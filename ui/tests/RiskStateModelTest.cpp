#include <QDateTime>
#include <QtTest>

#include "models/RiskStateModel.hpp"

class RiskStateModelTest : public QObject {
    Q_OBJECT

private slots:
    void updatesExposeData();
    void clearResetsModel();
};

void RiskStateModelTest::updatesExposeData() {
    RiskStateModel model;
    RiskSnapshotData snapshot;
    snapshot.profileLabel = QStringLiteral("Balanced");
    snapshot.portfolioValue = 1'250'000.0;
    snapshot.currentDrawdown = 0.015;
    snapshot.maxDailyLoss = 0.05;
    snapshot.usedLeverage = 1.3;
    snapshot.generatedAt = QDateTime::fromString(QStringLiteral("2024-03-01T12:00:00Z"), Qt::ISODate);
    snapshot.exposures.append({QStringLiteral("MAX_POSITION"), 100000.0, 90000.0, 95000.0});
    snapshot.exposures.append({QStringLiteral("PORTFOLIO_VAR"), 50000.0, 30000.0, 40000.0});

    model.updateFromSnapshot(snapshot);

    QCOMPARE(model.hasData(), true);
    QCOMPARE(model.profileLabel(), QStringLiteral("Balanced"));
    QCOMPARE(model.portfolioValue(), 1'250'000.0);
    QCOMPARE(model.currentDrawdown(), 0.015);
    QCOMPARE(model.maxDailyLoss(), 0.05);
    QCOMPARE(model.usedLeverage(), 1.3);
    QCOMPARE(model.generatedAt(), snapshot.generatedAt);
    QCOMPARE(model.rowCount(), 2);

    const QModelIndex first = model.index(0, 0);
    QCOMPARE(model.data(first, RiskStateModel::CodeRole).toString(), QStringLiteral("MAX_POSITION"));
    QCOMPARE(model.data(first, RiskStateModel::CurrentValueRole).toDouble(), 90000.0);
    QCOMPARE(model.data(first, RiskStateModel::ThresholdValueRole).toDouble(), 95000.0);
    QCOMPARE(model.data(first, RiskStateModel::BreachRole).toBool(), false);

    const QModelIndex second = model.index(1, 0);
    QCOMPARE(model.data(second, RiskStateModel::BreachRole).toBool(), false);
}

void RiskStateModelTest::clearResetsModel() {
    RiskStateModel model;
    RiskSnapshotData snapshot;
    snapshot.profileLabel = QStringLiteral("Aggressive");
    snapshot.exposures.append({QStringLiteral("LIMIT"), 10.0, 11.0, 9.0});
    model.updateFromSnapshot(snapshot);

    model.clear();

    QCOMPARE(model.hasData(), false);
    QCOMPARE(model.rowCount(), 0);
    QVERIFY(model.profileLabel().isEmpty());
}

QTEST_MAIN(RiskStateModelTest)
#include "RiskStateModelTest.moc"
