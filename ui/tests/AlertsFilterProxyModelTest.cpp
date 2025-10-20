#include <QtTest/QtTest>

#include "models/AlertsFilterProxyModel.hpp"
#include "models/AlertsModel.hpp"

class AlertsFilterProxyModelTest : public QObject {
    Q_OBJECT

private slots:
    void filtersBySeverity();
    void hidesAcknowledged();
    void filtersBySearchText();
    void sortsAlerts();
};

namespace {
RiskSnapshotData buildSnapshot()
{
    RiskSnapshotData snapshot;
    snapshot.currentDrawdown = 0.06; // warning
    snapshot.usedLeverage = 9.5;     // critical

    RiskExposureData warningExposure;
    warningExposure.code = QStringLiteral("MAX_POSITION");
    warningExposure.thresholdValue = 80000.0;
    warningExposure.currentValue = 70000.0; // warning

    RiskExposureData criticalExposure;
    criticalExposure.code = QStringLiteral("DAILY_LOSS");
    criticalExposure.thresholdValue = 50000.0;
    criticalExposure.currentValue = 52000.0; // critical breach

    snapshot.exposures.append(warningExposure);
    snapshot.exposures.append(criticalExposure);
    return snapshot;
}
} // namespace

void AlertsFilterProxyModelTest::filtersBySeverity()
{
    AlertsModel source;
    source.updateFromRiskSnapshot(buildSnapshot());
    QVERIFY(source.rowCount() >= 3);

    AlertsFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    proxy.setSeverityFilter(AlertsFilterProxyModel::AllSeverities);
    QCOMPARE(proxy.rowCount(), source.rowCount());

    proxy.setSeverityFilter(AlertsFilterProxyModel::CriticalOnly);
    QCOMPARE(proxy.rowCount(), source.criticalCount());

    proxy.setSeverityFilter(AlertsFilterProxyModel::WarningOnly);
    QCOMPARE(proxy.rowCount(), source.warningCount());

    proxy.setSeverityFilter(AlertsFilterProxyModel::WarningsAndCritical);
    QCOMPARE(proxy.rowCount(), source.warningCount() + source.criticalCount());
}

void AlertsFilterProxyModelTest::hidesAcknowledged()
{
    AlertsModel source;
    source.updateFromRiskSnapshot(buildSnapshot());

    AlertsFilterProxyModel proxy;
    proxy.setSourceModel(&source);
    proxy.setSeverityFilter(AlertsFilterProxyModel::WarningsAndCritical);

    const int initialCount = proxy.rowCount();
    QVERIFY(initialCount > 0);

    source.acknowledgeAll();
    QVERIFY(source.unacknowledgedCount() == 0);

    proxy.setHideAcknowledged(true);
    QCOMPARE(proxy.hideAcknowledged(), true);
    QCOMPARE(proxy.rowCount(), 0);

    proxy.setHideAcknowledged(false);
    QCOMPARE(proxy.rowCount(), initialCount);
}

void AlertsFilterProxyModelTest::filtersBySearchText()
{
    AlertsModel source;
    source.updateFromRiskSnapshot(buildSnapshot());

    AlertsFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    proxy.setSearchText(QStringLiteral("drawdown"));
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.index(0, 0).data(AlertsModel::IdRole).toString(), QStringLiteral("drawdown"));

    proxy.setSearchText(QStringLiteral("limit ekspozycji"));
    QCOMPARE(proxy.rowCount(), 2);

    proxy.setSearchText(QStringLiteral("dŹwigNIA"));
    QCOMPARE(proxy.rowCount(), 1);
    QCOMPARE(proxy.index(0, 0).data(AlertsModel::IdRole).toString(), QStringLiteral("leverage"));

    proxy.setSearchText(QString());
    QCOMPARE(proxy.rowCount(), source.rowCount());
}

void AlertsFilterProxyModelTest::sortsAlerts()
{
    AlertsModel source;

    auto exposure = [](const QString& code, double current, double threshold) {
        RiskExposureData data;
        data.code = code;
        data.currentValue = current;
        data.thresholdValue = threshold;
        return data;
    };

    RiskSnapshotData initial;
    initial.exposures.append(exposure(QStringLiteral("EXPA"), 85.0, 100.0)); // warning
    initial.exposures.append(exposure(QStringLiteral("EXPB"), 88.0, 100.0)); // warning

    source.updateFromRiskSnapshot(initial);
    QCOMPARE(source.rowCount(), 2);

    QTest::qSleep(5);

    RiskSnapshotData followup;
    followup.exposures.append(initial.exposures.at(0));
    auto criticalExposure = initial.exposures.at(1);
    criticalExposure.currentValue = 120.0; // critical
    followup.exposures.append(criticalExposure);

    source.updateFromRiskSnapshot(followup);
    QCOMPARE(source.rowCount(), 2);

    AlertsFilterProxyModel proxy;
    proxy.setSourceModel(&source);

    QCOMPARE(proxy.rowCount(), 2);

    QCOMPARE(proxy.index(0, 0).data(AlertsModel::IdRole).toString(), QStringLiteral("exposure:EXPB"));
    QCOMPARE(proxy.index(1, 0).data(AlertsModel::IdRole).toString(), QStringLiteral("exposure:EXPA"));

    proxy.setSortMode(AlertsFilterProxyModel::OldestFirst);
    QCOMPARE(proxy.index(0, 0).data(AlertsModel::IdRole).toString(), QStringLiteral("exposure:EXPA"));

    proxy.setSortMode(AlertsFilterProxyModel::SeverityDescending);
    QCOMPARE(proxy.index(0, 0).data(AlertsModel::SeverityRole).toInt(), AlertsModel::Critical);

    proxy.setSortMode(AlertsFilterProxyModel::SeverityAscending);
    QCOMPARE(proxy.index(0, 0).data(AlertsModel::SeverityRole).toInt(), AlertsModel::Warning);

    proxy.setSortMode(AlertsFilterProxyModel::TitleAscending);
    QCOMPARE(proxy.index(0, 0).data(AlertsModel::IdRole).toString(), QStringLiteral("exposure:EXPA"));
}

QTEST_MAIN(AlertsFilterProxyModelTest)
#include "AlertsFilterProxyModelTest.moc"
