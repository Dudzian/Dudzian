#include <QJsonArray>
#include <QJsonObject>
#include <QSignalSpy>
#include <QtTest/QtTest>

#include <cmath>

#include "models/RiskHistoryModel.hpp"

class RiskHistoryModelTest : public QObject {
    Q_OBJECT

private slots:
    void recordsSnapshots();
    void deduplicatesTimestamps();
    void trimsToMaximum();
    void updatesSummaryStatistics();
    void serializesAndRestoresJson();
    void capturesExposureDetails();
    void exposesEntryCountProperty();
};

void RiskHistoryModelTest::recordsSnapshots()
{
    RiskHistoryModel model;

    RiskSnapshotData snapshot;
    snapshot.hasData = true;
    snapshot.profileLabel = QStringLiteral("Balanced");
    snapshot.currentDrawdown = 0.0123;
    snapshot.usedLeverage = 1.45;
    snapshot.portfolioValue = 1'200'000.0;
    snapshot.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T10:15:00Z"), Qt::ISODate);

    model.recordSnapshot(snapshot);

    QCOMPARE(model.rowCount(), 1);
    QCOMPARE(model.hasSamples(), true);

    const QModelIndex index = model.index(0, 0);
    QCOMPARE(model.data(index, RiskHistoryModel::ProfileLabelRole).toString(), QStringLiteral("Balanced"));
    QCOMPARE(model.data(index, RiskHistoryModel::DrawdownRole).toDouble(), 0.0123);
    QCOMPARE(model.data(index, RiskHistoryModel::LeverageRole).toDouble(), 1.45);
    QCOMPARE(model.data(index, RiskHistoryModel::PortfolioValueRole).toDouble(), 1'200'000.0);
    QCOMPARE(model.data(index, RiskHistoryModel::TimestampRole).toDateTime(), snapshot.generatedAt.toUTC());
    QCOMPARE(model.data(index, RiskHistoryModel::BreachCountRole).toInt(), 0);
    QCOMPARE(model.data(index, RiskHistoryModel::HasBreachRole).toBool(), false);
    QVERIFY(model.data(index, RiskHistoryModel::ExposuresRole).toList().isEmpty());
}

void RiskHistoryModelTest::deduplicatesTimestamps()
{
    RiskHistoryModel model;

    RiskSnapshotData snapshot;
    snapshot.hasData = true;
    snapshot.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T11:00:00Z"), Qt::ISODate);
    snapshot.currentDrawdown = 0.02;
    snapshot.usedLeverage = 2.0;

    model.recordSnapshot(snapshot);

    RiskSnapshotData updated = snapshot;
    updated.currentDrawdown = 0.03;
    updated.usedLeverage = 1.8;
    updated.portfolioValue = 980'000.0;

    model.recordSnapshot(updated);

    QCOMPARE(model.rowCount(), 1);
    const QModelIndex index = model.index(0, 0);
    QCOMPARE(model.data(index, RiskHistoryModel::DrawdownRole).toDouble(), 0.03);
    QCOMPARE(model.data(index, RiskHistoryModel::LeverageRole).toDouble(), 1.8);
    QCOMPARE(model.data(index, RiskHistoryModel::PortfolioValueRole).toDouble(), 980'000.0);
}

void RiskHistoryModelTest::trimsToMaximum()
{
    RiskHistoryModel model;
    model.setMaximumEntries(2);

    RiskSnapshotData first;
    first.hasData = true;
    first.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T09:00:00Z"), Qt::ISODate);
    model.recordSnapshot(first);

    RiskSnapshotData second = first;
    second.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T09:01:00Z"), Qt::ISODate);
    second.currentDrawdown = 0.01;
    model.recordSnapshot(second);

    RiskSnapshotData third = first;
    third.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T09:02:00Z"), Qt::ISODate);
    third.currentDrawdown = 0.02;
    model.recordSnapshot(third);

    QCOMPARE(model.rowCount(), 2);
    const QModelIndex oldest = model.index(0, 0);
    QCOMPARE(model.data(oldest, RiskHistoryModel::TimestampRole).toDateTime(), second.generatedAt.toUTC());
}

void RiskHistoryModelTest::updatesSummaryStatistics()
{
    RiskHistoryModel model;

    QSignalSpy summarySpy(&model, &RiskHistoryModel::summaryChanged);

    RiskSnapshotData first;
    first.hasData = true;
    first.currentDrawdown = 0.05;
    first.usedLeverage = 1.2;
    first.portfolioValue = 1'000'000.0;
    first.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T09:00:00Z"), Qt::ISODate);

    RiskSnapshotData second = first;
    second.currentDrawdown = 0.15;
    second.usedLeverage = 2.5;
    second.portfolioValue = 1'250'000.0;
    second.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T09:01:00Z"), Qt::ISODate);

    RiskSnapshotData third = first;
    third.currentDrawdown = 0.08;
    third.usedLeverage = 1.8;
    third.portfolioValue = 900'000.0;
    third.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T09:02:00Z"), Qt::ISODate);

    model.recordSnapshot(first);
    QVERIFY(summarySpy.count() >= 1);
    summarySpy.clear();

    model.recordSnapshot(second);
    model.recordSnapshot(third);
    QVERIFY(summarySpy.count() >= 2);

    QCOMPARE(model.maxDrawdown(), 0.15);
    QCOMPARE(model.minDrawdown(), 0.05);
    QCOMPARE(model.maxLeverage(), 2.5);
    QVERIFY(std::abs(model.averageDrawdown() - ((0.05 + 0.15 + 0.08) / 3.0)) < 1e-9);
    QVERIFY(std::abs(model.averageLeverage() - ((1.2 + 2.5 + 1.8) / 3.0)) < 1e-9);
    QCOMPARE(model.minPortfolioValue(), 900'000.0);
    QCOMPARE(model.maxPortfolioValue(), 1'250'000.0);

    summarySpy.clear();
    model.clear();
    QVERIFY(summarySpy.count() >= 1);
    QCOMPARE(model.maxDrawdown(), 0.0);
    QCOMPARE(model.minDrawdown(), 0.0);
    QCOMPARE(model.averageDrawdown(), 0.0);
    QCOMPARE(model.maxLeverage(), 0.0);
    QCOMPARE(model.averageLeverage(), 0.0);
    QCOMPARE(model.minPortfolioValue(), 0.0);
    QCOMPARE(model.maxPortfolioValue(), 0.0);
}

void RiskHistoryModelTest::serializesAndRestoresJson()
{
    RiskHistoryModel model;

    RiskSnapshotData first;
    first.hasData = true;
    first.profileLabel = QStringLiteral("Alpha");
    first.currentDrawdown = 0.02;
    first.usedLeverage = 1.4;
    first.portfolioValue = 1'100'000.0;
    first.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T10:00:00Z"), Qt::ISODate);

    RiskSnapshotData second = first;
    second.profileLabel = QStringLiteral("Beta");
    second.currentDrawdown = 0.04;
    second.usedLeverage = 2.1;
    second.portfolioValue = 950'000.0;
    second.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T10:05:00Z"), Qt::ISODate);

    model.recordSnapshot(first);
    model.recordSnapshot(second);

    const QJsonArray fullJson = model.toJson();
    QCOMPARE(fullJson.size(), 2);
    const QString firstTimestamp = fullJson.at(0).toObject().value(QStringLiteral("timestamp")).toString();
    QVERIFY(!firstTimestamp.isEmpty());
    QVERIFY(QDateTime::fromString(firstTimestamp, Qt::ISODateWithMs).isValid());

    const QJsonArray limitedJson = model.toJson(1);
    QCOMPARE(limitedJson.size(), 1);
    const QJsonObject limitedObject = limitedJson.at(0).toObject();
    QCOMPARE(limitedObject.value(QStringLiteral("profileLabel")).toString(), QStringLiteral("Beta"));

    RiskHistoryModel restored;
    restored.restoreFromJson(fullJson);
    QCOMPARE(restored.rowCount(), 2);
    const QModelIndex restoredFirst = restored.index(0, 0);
    QCOMPARE(restored.data(restoredFirst, RiskHistoryModel::ProfileLabelRole).toString(), QStringLiteral("Alpha"));
    const QModelIndex restoredSecond = restored.index(1, 0);
    QCOMPARE(restored.data(restoredSecond, RiskHistoryModel::ProfileLabelRole).toString(), QStringLiteral("Beta"));
    QVERIFY(std::abs(restored.maxDrawdown() - 0.04) < 1e-9);
    QVERIFY(std::abs(restored.minDrawdown() - 0.02) < 1e-9);
    QCOMPARE(restored.totalBreachCount(), 0);

    restored.setMaximumEntries(1);
    restored.restoreFromJson(fullJson);
    QCOMPARE(restored.rowCount(), 1);
    QCOMPARE(restored.data(restored.index(0, 0), RiskHistoryModel::ProfileLabelRole).toString(), QStringLiteral("Beta"));
}

void RiskHistoryModelTest::capturesExposureDetails()
{
    RiskHistoryModel model;

    RiskSnapshotData snapshot;
    snapshot.hasData = true;
    snapshot.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T13:00:00Z"), Qt::ISODate);
    snapshot.currentDrawdown = 0.12;
    snapshot.usedLeverage = 2.4;

    RiskExposureData stableExposure;
    stableExposure.code = QStringLiteral("FUNDING");
    stableExposure.currentValue = 45'000.0;
    stableExposure.thresholdValue = 100'000.0;
    stableExposure.maxValue = 125'000.0;

    RiskExposureData breachedExposure;
    breachedExposure.code = QStringLiteral("DRAWDOWN");
    breachedExposure.currentValue = 110'000.0;
    breachedExposure.thresholdValue = 100'000.0;
    breachedExposure.maxValue = 120'000.0;

    snapshot.exposures = {stableExposure, breachedExposure};

    model.recordSnapshot(snapshot);

    QCOMPARE(model.totalBreachCount(), 1);
    QVERIFY(model.anyExposureBreached());
    QVERIFY(model.maxExposureUtilization() > 1.0);

    const QModelIndex index = model.index(0, 0);
    QCOMPARE(model.data(index, RiskHistoryModel::HasBreachRole).toBool(), true);
    QCOMPARE(model.data(index, RiskHistoryModel::BreachCountRole).toInt(), 1);

    const QVariantList exposures = model.data(index, RiskHistoryModel::ExposuresRole).toList();
    QCOMPARE(exposures.size(), 2);
    const QVariantMap lastExposure = exposures.at(1).toMap();
    QCOMPARE(lastExposure.value(QStringLiteral("code")).toString(), QStringLiteral("DRAWDOWN"));
    QVERIFY(lastExposure.value(QStringLiteral("breached")).toBool());
    QVERIFY(lastExposure.value(QStringLiteral("utilization")).toDouble() > 1.0);

    const QJsonArray json = model.toJson();
    QCOMPARE(json.size(), 1);
    const QJsonObject jsonEntry = json.at(0).toObject();
    const QJsonArray exposuresArray = jsonEntry.value(QStringLiteral("exposures")).toArray();
    QCOMPARE(exposuresArray.size(), 2);
    QVERIFY(exposuresArray.at(1).toObject().value(QStringLiteral("breached")).toBool());

    RiskHistoryModel restored;
    restored.restoreFromJson(json);
    QCOMPARE(restored.rowCount(), 1);
    QCOMPARE(restored.totalBreachCount(), 1);
    QVERIFY(restored.anyExposureBreached());
    QVERIFY(restored.maxExposureUtilization() > 1.0);
}

void RiskHistoryModelTest::exposesEntryCountProperty()
{
    RiskHistoryModel model;
    QCOMPARE(model.entryCount(), 0);

    RiskSnapshotData snapshot;
    snapshot.hasData = true;
    snapshot.generatedAt = QDateTime::currentDateTimeUtc();
    model.recordSnapshot(snapshot);

    QCOMPARE(model.entryCount(), 1);

    model.clear();
    QCOMPARE(model.entryCount(), 0);
}

QTEST_MAIN(RiskHistoryModelTest)
#include "RiskHistoryModelTest.moc"

