#include <QVector>
#include <QtTest/QtTest>

#include "models/MarketRegimeTimelineModel.hpp"

class MarketRegimeTimelineModelTest : public QObject {
    Q_OBJECT

private slots:
    void trimsOnAppend();
    void trimsOnResetAndLimitChange();
    void unlimitedWhenZero();
};

namespace {
MarketRegimeSnapshotEntry makeSnapshot(int index)
{
    MarketRegimeSnapshotEntry entry;
    entry.timestampMs = static_cast<qint64>(index) * 1000;
    entry.regime = QStringLiteral("regime_%1").arg(index);
    entry.trendConfidence = 0.1 * index;
    entry.meanReversionConfidence = 0.2 * index;
    entry.dailyConfidence = 0.3 * index;
    return entry;
}
} // namespace

void MarketRegimeTimelineModelTest::trimsOnAppend()
{
    MarketRegimeTimelineModel model;
    model.setMaximumSnapshots(3);

    for (int i = 0; i < 5; ++i)
        model.appendSnapshot(makeSnapshot(i));

    QCOMPARE(model.rowCount(), 3);
    const QModelIndex first = model.index(0, 0);
    QCOMPARE(model.data(first, MarketRegimeTimelineModel::TimestampRole).toLongLong(), 2000LL);
    QCOMPARE(model.latestRegime(), QStringLiteral("regime_4"));
}

void MarketRegimeTimelineModelTest::trimsOnResetAndLimitChange()
{
    MarketRegimeTimelineModel model;
    model.setMaximumSnapshots(4);

    QVector<MarketRegimeSnapshotEntry> snapshots;
    for (int i = 0; i < 6; ++i)
        snapshots.append(makeSnapshot(i));

    model.resetWithSnapshots(snapshots);
    QCOMPARE(model.rowCount(), 4);
    QCOMPARE(model.data(model.index(0, 0), MarketRegimeTimelineModel::TimestampRole).toLongLong(), 2000LL);
    QCOMPARE(model.latestRegime(), QStringLiteral("regime_5"));

    model.setMaximumSnapshots(2);
    QCOMPARE(model.rowCount(), 2);
    QCOMPARE(model.data(model.index(0, 0), MarketRegimeTimelineModel::TimestampRole).toLongLong(), 4000LL);
    QCOMPARE(model.latestRegime(), QStringLiteral("regime_5"));
}

void MarketRegimeTimelineModelTest::unlimitedWhenZero()
{
    MarketRegimeTimelineModel model;
    model.setMaximumSnapshots(0);

    for (int i = 0; i < 10; ++i)
        model.appendSnapshot(makeSnapshot(i));

    QCOMPARE(model.rowCount(), 10);
    QCOMPARE(model.latestRegime(), QStringLiteral("regime_9"));
}

QTEST_MAIN(MarketRegimeTimelineModelTest)
#include "MarketRegimeTimelineModelTest.moc"
