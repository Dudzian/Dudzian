#include <QtTest/QtTest>
#include <QTemporaryDir>
#include <QFile>
#include <QFileInfo>
#include <QDir>

#include "app/StrategyConfigController.hpp"

class StrategyConfigControllerTest : public QObject {
    Q_OBJECT

private:
    QTemporaryDir m_tempDir;
    QString m_tempConfigPath;

private slots:
    void initTestCase();
    void test_decision_config_roundtrip();
    void test_scheduler_config_roundtrip();
};

void StrategyConfigControllerTest::initTestCase()
{
    QVERIFY2(m_tempDir.isValid(), "Nie udało się utworzyć katalogu tymczasowego dla testów");
    const QString sourceConfig = QDir::current().absoluteFilePath(QStringLiteral("config/core.yaml"));
    QVERIFY2(QFile::exists(sourceConfig), "Brak pliku config/core.yaml w katalogu roboczym testu");

    m_tempConfigPath = QDir(m_tempDir.path()).absoluteFilePath(QStringLiteral("core.yaml"));
    QVERIFY2(QFile::copy(sourceConfig, m_tempConfigPath), "Nie udało się skopiować config/core.yaml do katalogu tymczasowego");
}

void StrategyConfigControllerTest::test_decision_config_roundtrip()
{
    StrategyConfigController controller;
    controller.setConfigPath(m_tempConfigPath);
    controller.setScriptPath(QDir::current().absoluteFilePath(QStringLiteral("scripts/ui_config_bridge.py")));
    controller.setPythonExecutable(QStringLiteral("python3"));

    QVERIFY(controller.refresh());
    QVariantMap decision = controller.decisionConfigSnapshot();
    QVERIFY(!decision.isEmpty());

    const double originalMaxCost = decision.value(QStringLiteral("max_cost_bps")).toDouble();
    decision.insert(QStringLiteral("max_cost_bps"), originalMaxCost + 1.25);

    QVariantList overrides = decision.value(QStringLiteral("profile_overrides")).toList();
    if (!overrides.isEmpty()) {
        QVariantMap firstOverride = overrides.first().toMap();
        firstOverride.insert(QStringLiteral("max_latency_ms"), firstOverride.value(QStringLiteral("max_latency_ms")).toDouble() + 10.0);
        overrides[0] = firstOverride;
        decision.insert(QStringLiteral("profile_overrides"), overrides);
    }

    QVERIFY(controller.saveDecisionConfig(decision));

    QVariantMap updated = controller.decisionConfigSnapshot();
    QCOMPARE(updated.value(QStringLiteral("max_cost_bps")).toDouble(), originalMaxCost + 1.25);
    if (!overrides.isEmpty()) {
        const QVariantList updatedOverrides = updated.value(QStringLiteral("profile_overrides")).toList();
        QVERIFY(!updatedOverrides.isEmpty());
        const QVariantMap updatedFirst = updatedOverrides.first().toMap();
        QVERIFY(updatedFirst.value(QStringLiteral("max_latency_ms")).toDouble() >= overrides.first().toMap().value(QStringLiteral("max_latency_ms")).toDouble());
    }
}

void StrategyConfigControllerTest::test_scheduler_config_roundtrip()
{
    StrategyConfigController controller;
    controller.setConfigPath(m_tempConfigPath);
    controller.setScriptPath(QDir::current().absoluteFilePath(QStringLiteral("scripts/ui_config_bridge.py")));
    controller.setPythonExecutable(QStringLiteral("python3"));

    QVERIFY(controller.refresh());
    const QVariantList schedulers = controller.schedulerList();
    QVERIFY(!schedulers.isEmpty());

    const QString schedulerName = schedulers.first().toMap().value(QStringLiteral("name")).toString();
    QVERIFY(!schedulerName.isEmpty());

    QVariantMap schedulerConfig = controller.schedulerConfigSnapshot(schedulerName);
    QVERIFY(!schedulerConfig.isEmpty());

    const int originalHealth = schedulerConfig.value(QStringLiteral("health_check_interval")).toInt();
    schedulerConfig.insert(QStringLiteral("health_check_interval"), originalHealth + 15);

    QVariantList schedules = schedulerConfig.value(QStringLiteral("schedules")).toList();
    QVERIFY(!schedules.isEmpty());

    QVariantMap firstSchedule = schedules.first().toMap();
    const int originalSignals = firstSchedule.value(QStringLiteral("max_signals")).toInt();
    firstSchedule.insert(QStringLiteral("max_signals"), originalSignals + 2);
    schedules[0] = firstSchedule;
    schedulerConfig.insert(QStringLiteral("schedules"), schedules);

    QVERIFY(controller.saveSchedulerConfig(schedulerName, schedulerConfig));

    QVariantMap updated = controller.schedulerConfigSnapshot(schedulerName);
    QCOMPARE(updated.value(QStringLiteral("health_check_interval")).toInt(), originalHealth + 15);

    const QVariantList updatedSchedules = updated.value(QStringLiteral("schedules")).toList();
    QVERIFY(!updatedSchedules.isEmpty());
    QCOMPARE(updatedSchedules.first().toMap().value(QStringLiteral("max_signals")).toInt(), originalSignals + 2);
}

QTEST_MAIN(StrategyConfigControllerTest)
#include "StrategyConfigControllerTest.moc"
