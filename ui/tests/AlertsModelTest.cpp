#include <QtTest/QtTest>
#include <QStringList>

#include "models/AlertsModel.hpp"

class AlertsModelTest : public QObject {
    Q_OBJECT

private slots:
    void generatesAlertsFromRiskSnapshot();
    void acknowledgesAndClearsAlerts();
    void persistsAcknowledgedAlerts();
    void acknowledgeAllMarksAlerts();
};

void AlertsModelTest::generatesAlertsFromRiskSnapshot()
{
    AlertsModel model;
    RiskSnapshotData snapshot;
    snapshot.currentDrawdown = 0.06; // warning
    snapshot.usedLeverage = 9.0;      // critical

    RiskExposureData warningExposure;
    warningExposure.code = QStringLiteral("DAILY_LOSS");
    warningExposure.maxValue = 100000.0;
    warningExposure.thresholdValue = 80000.0;
    warningExposure.currentValue = 70000.0; // warning

    RiskExposureData criticalExposure;
    criticalExposure.code = QStringLiteral("POSITION_SIZE");
    criticalExposure.maxValue = 50000.0;
    criticalExposure.thresholdValue = 40000.0;
    criticalExposure.currentValue = 45000.0; // breach

    snapshot.exposures.append(warningExposure);
    snapshot.exposures.append(criticalExposure);

    model.updateFromRiskSnapshot(snapshot);

    QCOMPARE(model.rowCount(), 4); // drawdown, leverage, warning exposure, critical exposure
    QCOMPARE(model.warningCount(), 2); // drawdown + warning exposure
    QCOMPARE(model.criticalCount(), 2); // leverage + critical exposure

    QModelIndex first = model.index(0, 0);
    QVERIFY(first.isValid());
    QVERIFY(!model.data(first, AlertsModel::AcknowledgedRole).toBool());
}

void AlertsModelTest::acknowledgesAndClearsAlerts()
{
    AlertsModel model;
    RiskSnapshotData snapshot;
    snapshot.currentDrawdown = 0.09; // critical
    model.updateFromRiskSnapshot(snapshot);

    QCOMPARE(model.criticalCount(), 1);
    QModelIndex idx = model.index(0, 0);
    const QString alertId = model.data(idx, AlertsModel::IdRole).toString();
    QVERIFY(!alertId.isEmpty());

    model.acknowledge(alertId);
    QVERIFY(model.data(idx, AlertsModel::AcknowledgedRole).toBool());

    model.clearAcknowledged();
    QCOMPARE(model.rowCount(), 0);
    QCOMPARE(model.hasActiveAlerts(), false);
}

void AlertsModelTest::persistsAcknowledgedAlerts()
{
    AlertsModel model;
    RiskSnapshotData snapshot;
    snapshot.currentDrawdown = 0.06; // warning
    model.updateFromRiskSnapshot(snapshot);

    QVERIFY(model.rowCount() > 0);
    const QString alertId = model.data(model.index(0, 0), AlertsModel::IdRole).toString();
    QVERIFY(!alertId.isEmpty());

    model.acknowledge(alertId);
    QStringList acknowledged = model.acknowledgedAlertIds();
    QCOMPARE(acknowledged.size(), 1);
    QCOMPARE(acknowledged.first(), alertId);

    AlertsModel reloaded;
    reloaded.setAcknowledgedAlertIds(acknowledged);
    reloaded.updateFromRiskSnapshot(snapshot);
    int foundRow = -1;
    for (int row = 0; row < reloaded.rowCount(); ++row) {
        if (reloaded.data(reloaded.index(row, 0), AlertsModel::IdRole).toString() == alertId) {
            foundRow = row;
            break;
        }
    }
    QVERIFY(foundRow >= 0);
    QCOMPARE(reloaded.data(reloaded.index(foundRow, 0), AlertsModel::AcknowledgedRole).toBool(), true);

    snapshot.currentDrawdown = 0.12; // severity change to critical resets acknowledgement
    model.updateFromRiskSnapshot(snapshot);
    acknowledged = model.acknowledgedAlertIds();
    QCOMPARE(acknowledged.size(), 0);
    bool found = false;
    for (int row = 0; row < model.rowCount(); ++row) {
        if (model.data(model.index(row, 0), AlertsModel::IdRole).toString() == alertId) {
            QCOMPARE(model.data(model.index(row, 0), AlertsModel::AcknowledgedRole).toBool(), false);
            found = true;
            break;
        }
    }
    QVERIFY(found);
}

void AlertsModelTest::acknowledgeAllMarksAlerts()
{
    AlertsModel model;
    RiskSnapshotData snapshot;
    snapshot.currentDrawdown = 0.06; // warning
    snapshot.usedLeverage = 9.5;     // critical

    RiskExposureData exposure;
    exposure.code = QStringLiteral("POSITION_LIMIT");
    exposure.maxValue = 100000.0;
    exposure.thresholdValue = 80000.0;
    exposure.currentValue = 90000.0; // breach -> critical
    snapshot.exposures.append(exposure);

    model.updateFromRiskSnapshot(snapshot);

    QCOMPARE(model.rowCount(), 3);
    QCOMPARE(model.unacknowledgedCount(), 3);
    QVERIFY(model.hasUnacknowledgedAlerts());

    const QString firstId = model.data(model.index(0, 0), AlertsModel::IdRole).toString();
    QVERIFY(!firstId.isEmpty());
    model.acknowledge(firstId);
    QCOMPARE(model.unacknowledgedCount(), 2);
    QVERIFY(model.hasUnacknowledgedAlerts());

    model.acknowledgeAll();
    QCOMPARE(model.unacknowledgedCount(), 0);
    QVERIFY(!model.hasUnacknowledgedAlerts());
    QCOMPARE(model.acknowledgedAlertIds().size(), model.rowCount());

    model.acknowledgeAll(); // idempotent
    QCOMPARE(model.unacknowledgedCount(), 0);
}

QTEST_MAIN(AlertsModelTest)
#include "AlertsModelTest.moc"
