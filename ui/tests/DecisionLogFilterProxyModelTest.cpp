#include <QtTest>
#include <QFile>
#include <QTemporaryDir>
#include <QUrl>

#include "models/DecisionLogFilterProxyModel.hpp"

namespace {
class FakeDecisionModel : public QAbstractListModel {
    Q_OBJECT
public:
    struct Entry {
        QString timestamp;
        QString strategy;
        QString symbol;
        QString side;
        QString decisionState;
        QString decisionReason;
        bool approved = false;
        QString decisionMode;
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
        case DecisionLogModel::TimestampDisplayRole:
            return entry.timestamp;
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
            return QStringLiteral("order");
        case DecisionLogModel::TelemetryNamespaceRole:
            return QStringLiteral("trend");
        default:
            return {};
        }
    }

    QHash<int, QByteArray> roleNames() const override {
        return {
            {DecisionLogModel::TimestampDisplayRole, "timestampDisplay"},
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
};

void DecisionLogFilterProxyModelTest::filtersBySearchText() {
    FakeDecisionModel source;
    QVector<FakeDecisionModel::Entry> entries = {
        {"2024-01-01T10:00", "trend", "BTC/USDT", "BUY", "approved", "Breakout", true, "auto"},
        {"2024-01-01T10:05", "mean_reversion", "BTC/USDT", "SELL", "pending", "Rebalance", false, "manual"}
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
        {"2024-01-01T10:00", "trend", "BTC/USDT", "BUY", "approved", "Breakout", true, "auto"},
        {"2024-01-01T10:05", "mean_reversion", "BTC/USDT", "SELL", "pending", "Rebalance", false, "manual"}
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

QTEST_MAIN(DecisionLogFilterProxyModelTest)
#include "DecisionLogFilterProxyModelTest.moc"
