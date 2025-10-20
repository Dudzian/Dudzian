#include <QtTest/QtTest>
#include <cmath>
#include <QByteArray>
#include <QCommandLineParser>
#include <QFileInfo>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStringList>
#include <QQmlApplicationEngine>
#include <QTemporaryDir>

#include "app/Application.hpp"
#include "models/AlertsModel.hpp"
#include "models/AlertsFilterProxyModel.hpp"
#include "models/RiskTypes.hpp"
#include "models/RiskHistoryModel.hpp"

namespace {

struct EnvRestore {
    QByteArray key;
    QByteArray value;
    bool hadValue = false;

    explicit EnvRestore(QByteArray k)
        : key(std::move(k))
        , value(qgetenv(key.constData()))
        , hadValue(qEnvironmentVariableIsSet(key.constData()))
    {
    }

    ~EnvRestore()
    {
        if (hadValue)
            qputenv(key.constData(), value);
        else
            qunsetenv(key.constData());
    }
};

} // namespace

class ApplicationSettingsPersistenceTest : public QObject {
    Q_OBJECT

private slots:
    void testPersistsAndReloadsConfiguration();
    void testDisableUiSettingsSkipsWrites();
    void testCliOverridesUiSettingsPath();
};

void ApplicationSettingsPersistenceTest::testPersistsAndReloadsConfiguration()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString settingsPath = dir.filePath(QStringLiteral("ui_settings.json"));

    constexpr auto kEnvName = QByteArrayLiteral("BOT_CORE_UI_SETTINGS_PATH");
    EnvRestore guard(kEnvName);
    qputenv(kEnvName.constData(), settingsPath.toUtf8());

    {
        QQmlApplicationEngine engine;
        Application app(engine);

        QVERIFY(app.updateInstrument(QStringLiteral("TESTX"),
                                     QStringLiteral("FOO/BAR"),
                                     QStringLiteral("FOOBAR"),
                                     QStringLiteral("BAR"),
                                     QStringLiteral("FOO"),
                                     QStringLiteral("PT5M")));
        QVERIFY(app.updatePerformanceGuard(75, 1.25, 14.5, 5, 42));
        QVERIFY(app.updateRiskRefresh(true, 7.5));

        auto* alertsModel = qobject_cast<AlertsModel*>(app.alertsModel());
        QVERIFY(alertsModel);
        RiskSnapshotData snapshot;
        snapshot.currentDrawdown = 0.06; // warning alert
        alertsModel->updateFromRiskSnapshot(snapshot);
        QVERIFY(alertsModel->rowCount() > 0);
        const QString alertId = alertsModel->data(alertsModel->index(0, 0), AlertsModel::IdRole).toString();
        alertsModel->acknowledge(alertId);

        auto* filterModel = qobject_cast<AlertsFilterProxyModel*>(app.alertsFilterModel());
        QVERIFY(filterModel);
        filterModel->setHideAcknowledged(true);
        filterModel->setSeverityFilter(AlertsFilterProxyModel::CriticalOnly);
        filterModel->setSortMode(AlertsFilterProxyModel::TitleAscending);
        filterModel->setSearchText(QStringLiteral("drawdown"));

        auto* historyModel = qobject_cast<RiskHistoryModel*>(app.riskHistoryModel());
        QVERIFY(historyModel);
        historyModel->setMaximumEntries(75);
        RiskSnapshotData historySample;
        historySample.hasData = true;
        historySample.profileLabel = QStringLiteral("Profil testowy");
        historySample.currentDrawdown = 0.041;
        historySample.usedLeverage = 1.75;
        historySample.portfolioValue = 875000.0;
        historySample.generatedAt = QDateTime::fromString(QStringLiteral("2024-04-01T12:00:00Z"), Qt::ISODate);
        RiskExposureData exposureA;
        exposureA.code = QStringLiteral("LEVERAGE");
        exposureA.currentValue = 80'000.0;
        exposureA.thresholdValue = 120'000.0;
        exposureA.maxValue = 150'000.0;
        RiskExposureData exposureB;
        exposureB.code = QStringLiteral("DRAWDOWN");
        exposureB.currentValue = 130'000.0;
        exposureB.thresholdValue = 125'000.0;
        exposureB.maxValue = 140'000.0;
        historySample.exposures = {exposureA, exposureB};
        historyModel->recordSnapshot(historySample);

        app.saveUiSettingsImmediatelyForTesting();
        QVERIFY(QFile::exists(settingsPath));

        QFile file(settingsPath);
        QVERIFY(file.open(QIODevice::ReadOnly | QIODevice::Text));
        const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
        QVERIFY(doc.isObject());
        const QJsonObject root = doc.object();
        QVERIFY(root.value(QStringLiteral("instrument")).isObject());
        QVERIFY(root.value(QStringLiteral("performanceGuard")).isObject());
        QVERIFY(root.value(QStringLiteral("riskRefresh")).isObject());
        QVERIFY(root.value(QStringLiteral("alerts")).isObject());
        QVERIFY(root.value(QStringLiteral("riskHistory")).isObject());
        const QJsonObject alertsSection = root.value(QStringLiteral("alerts")).toObject();
        const QJsonArray acknowledged = alertsSection.value(QStringLiteral("acknowledgedIds")).toArray();
        QCOMPARE(acknowledged.size(), 1);
        QCOMPARE(acknowledged.at(0).toString(), alertId);
        QVERIFY(alertsSection.contains(QStringLiteral("hideAcknowledged")));
        QVERIFY(alertsSection.value(QStringLiteral("hideAcknowledged")).toBool());
        QCOMPARE(alertsSection.value(QStringLiteral("severityFilter")).toInt(),
                 static_cast<int>(AlertsFilterProxyModel::CriticalOnly));
        QCOMPARE(alertsSection.value(QStringLiteral("sortMode")).toInt(),
                 static_cast<int>(AlertsFilterProxyModel::TitleAscending));
        QCOMPARE(alertsSection.value(QStringLiteral("searchText")).toString(), QStringLiteral("drawdown"));

        const QJsonObject historySection = root.value(QStringLiteral("riskHistory")).toObject();
        QCOMPARE(historySection.value(QStringLiteral("maximumEntries")).toInt(), 75);
        const QJsonArray historyArray = historySection.value(QStringLiteral("entries")).toArray();
        QCOMPARE(historyArray.size(), 1);
        const QJsonObject historyObject = historyArray.at(0).toObject();
        QCOMPARE(historyObject.value(QStringLiteral("profileLabel")).toString(), QStringLiteral("Profil testowy"));
        QVERIFY(std::abs(historyObject.value(QStringLiteral("drawdown")).toDouble() - historySample.currentDrawdown)
                < 1e-9);
        QVERIFY(std::abs(historyObject.value(QStringLiteral("leverage")).toDouble() - historySample.usedLeverage)
                < 1e-9);
        QVERIFY(std::abs(historyObject.value(QStringLiteral("portfolioValue")).toDouble()
                         - historySample.portfolioValue)
                < 1e-6);
        const QString timestamp = historyObject.value(QStringLiteral("timestamp")).toString();
        QVERIFY(!timestamp.isEmpty());
        const QDateTime parsedTimestamp = QDateTime::fromString(timestamp, Qt::ISODateWithMs);
        QVERIFY(parsedTimestamp.isValid());
        const QJsonArray persistedExposures = historyObject.value(QStringLiteral("exposures")).toArray();
        QCOMPARE(persistedExposures.size(), 2);
        QVERIFY(persistedExposures.at(1).toObject().value(QStringLiteral("breached")).toBool());
    }

    {
        QQmlApplicationEngine engine;
        Application app(engine);

        const auto instrument = app.instrumentConfigSnapshot();
        QCOMPARE(instrument.value(QStringLiteral("exchange")).toString(), QStringLiteral("TESTX"));
        QCOMPARE(instrument.value(QStringLiteral("symbol")).toString(), QStringLiteral("FOO/BAR"));
        QCOMPARE(instrument.value(QStringLiteral("granularity")).toString(), QStringLiteral("PT5M"));

        const auto guardSnapshot = app.performanceGuardSnapshot();
        QCOMPARE(guardSnapshot.value(QStringLiteral("fpsTarget")).toInt(), 75);
        QCOMPARE(guardSnapshot.value(QStringLiteral("maxOverlayCount")).toInt(), 5);
        QCOMPARE(guardSnapshot.value(QStringLiteral("disableSecondaryWhenBelow")).toInt(), 42);
        QCOMPARE(guardSnapshot.value(QStringLiteral("reduceMotionAfter")).toDouble(), 1.25);
        QCOMPARE(guardSnapshot.value(QStringLiteral("jankThresholdMs")).toDouble(), 14.5);

        const auto riskSnapshot = app.riskRefreshSnapshot();
        QVERIFY(riskSnapshot.value(QStringLiteral("enabled")).toBool());
        QCOMPARE(riskSnapshot.value(QStringLiteral("intervalSeconds")).toDouble(), 7.5);

        auto* alertsModel = qobject_cast<AlertsModel*>(app.alertsModel());
        QVERIFY(alertsModel);
        RiskSnapshotData snapshot;
        snapshot.currentDrawdown = 0.06;
        alertsModel->updateFromRiskSnapshot(snapshot);
        bool foundAcknowledged = false;
        for (int row = 0; row < alertsModel->rowCount(); ++row) {
            if (alertsModel->data(alertsModel->index(row, 0), AlertsModel::AcknowledgedRole).toBool()) {
                foundAcknowledged = true;
                break;
            }
        }
        QVERIFY(foundAcknowledged);

        auto* filterModel = qobject_cast<AlertsFilterProxyModel*>(app.alertsFilterModel());
        QVERIFY(filterModel);
        QVERIFY(filterModel->hideAcknowledged());
        QCOMPARE(filterModel->severityFilter(), AlertsFilterProxyModel::CriticalOnly);
        QCOMPARE(filterModel->sortMode(), AlertsFilterProxyModel::TitleAscending);
        QCOMPARE(filterModel->searchText(), QStringLiteral("drawdown"));

        auto* historyModel = qobject_cast<RiskHistoryModel*>(app.riskHistoryModel());
        QVERIFY(historyModel);
        QCOMPARE(historyModel->maximumEntries(), 75);
        QCOMPARE(historyModel->rowCount(), 1);
        const QModelIndex row = historyModel->index(0, 0);
        QCOMPARE(historyModel->data(row, RiskHistoryModel::ProfileLabelRole).toString(),
                 QStringLiteral("Profil testowy"));
        QVERIFY(std::abs(historyModel->data(row, RiskHistoryModel::DrawdownRole).toDouble() - 0.041) < 1e-9);
        QVERIFY(std::abs(historyModel->data(row, RiskHistoryModel::LeverageRole).toDouble() - 1.75) < 1e-9);
        QVERIFY(std::abs(historyModel->data(row, RiskHistoryModel::PortfolioValueRole).toDouble() - 875000.0)
                < 1e-3);
        QVERIFY(historyModel->data(row, RiskHistoryModel::TimestampRole).toDateTime().isValid());
        const QVariantList restoredExposures = historyModel->data(row, RiskHistoryModel::ExposuresRole).toList();
        QCOMPARE(restoredExposures.size(), 2);
        QVERIFY(restoredExposures.at(1).toMap().value(QStringLiteral("breached")).toBool());
        QCOMPARE(historyModel->maxDrawdown(), 0.041);
        QCOMPARE(historyModel->minDrawdown(), 0.041);
        QCOMPARE(historyModel->totalBreachCount(), 1);
        QVERIFY(historyModel->anyExposureBreached());
        QVERIFY(historyModel->maxExposureUtilization() > 1.0);
    }
}

void ApplicationSettingsPersistenceTest::testDisableUiSettingsSkipsWrites()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString settingsPath = dir.filePath(QStringLiteral("ui_disabled.json"));

    constexpr auto kEnvName = QByteArrayLiteral("BOT_CORE_UI_SETTINGS_PATH");
    constexpr auto kDisableEnv = QByteArrayLiteral("BOT_CORE_UI_SETTINGS_DISABLE");
    EnvRestore pathGuard(kEnvName);
    EnvRestore disableGuard(kDisableEnv);
    qputenv(kEnvName.constData(), settingsPath.toUtf8());
    qputenv(kDisableEnv.constData(), QByteArrayLiteral("1"));

    QQmlApplicationEngine engine;
    Application app(engine);

    QCOMPARE(app.uiSettingsPathForTesting(), QFileInfo(settingsPath).absoluteFilePath());
    QVERIFY(!app.uiSettingsPersistenceEnabledForTesting());

    QVERIFY(app.updateInstrument(QStringLiteral("DISABLED"),
                                 QStringLiteral("AAA/BBB"),
                                 QStringLiteral("AAABBB"),
                                 QStringLiteral("BBB"),
                                 QStringLiteral("AAA"),
                                 QStringLiteral("PT15M")));

    app.saveUiSettingsImmediatelyForTesting();
    QVERIFY(!QFile::exists(settingsPath));
}

void ApplicationSettingsPersistenceTest::testCliOverridesUiSettingsPath()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString overridePath = dir.filePath(QStringLiteral("cli_override.json"));

    QJsonObject seed;
    QJsonObject instrument{{QStringLiteral("exchange"), QStringLiteral("CLIEX")},
                           {QStringLiteral("symbol"), QStringLiteral("FOO/BAR")},
                           {QStringLiteral("venueSymbol"), QStringLiteral("FOOBAR")},
                           {QStringLiteral("quoteCurrency"), QStringLiteral("BAR")},
                           {QStringLiteral("baseCurrency"), QStringLiteral("FOO")},
                           {QStringLiteral("granularity"), QStringLiteral("PT10M")}};
    QJsonObject guard{{QStringLiteral("fpsTarget"), 90},
                      {QStringLiteral("reduceMotionAfter"), 2.5},
                      {QStringLiteral("jankThresholdMs"), 12.0},
                      {QStringLiteral("maxOverlayCount"), 6},
                      {QStringLiteral("disableSecondaryWhenBelow"), 33}};
    QJsonObject risk{{QStringLiteral("enabled"), false},
                     {QStringLiteral("intervalSeconds"), 42.0}};
    seed.insert(QStringLiteral("instrument"), instrument);
    seed.insert(QStringLiteral("performanceGuard"), guard);
    seed.insert(QStringLiteral("riskRefresh"), risk);

    {
        QFile file(overridePath);
        QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));
        QJsonDocument doc(seed);
        file.write(doc.toJson(QJsonDocument::Compact));
        file.close();
    }

    constexpr auto kDisableEnv = QByteArrayLiteral("BOT_CORE_UI_SETTINGS_DISABLE");
    EnvRestore disableGuard(kDisableEnv);
    qputenv(kDisableEnv.constData(), QByteArrayLiteral("1"));

    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args = {QStringLiteral("app"),
                              QStringLiteral("--enable-ui-settings"),
                              QStringLiteral("--ui-settings-path"),
                              overridePath};
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    QVERIFY(app.uiSettingsPersistenceEnabledForTesting());
    QCOMPARE(app.uiSettingsPathForTesting(), QFileInfo(overridePath).absoluteFilePath());

    const auto instrumentSnapshot = app.instrumentConfigSnapshot();
    QCOMPARE(instrumentSnapshot.value(QStringLiteral("exchange")).toString(), QStringLiteral("CLIEX"));
    QCOMPARE(instrumentSnapshot.value(QStringLiteral("granularity")).toString(), QStringLiteral("PT10M"));

    const auto riskSnapshot = app.riskRefreshSnapshot();
    QVERIFY(!riskSnapshot.value(QStringLiteral("enabled")).toBool());
    QCOMPARE(riskSnapshot.value(QStringLiteral("intervalSeconds")).toDouble(), 42.0);

    QVERIFY(app.updateRiskRefresh(true, 9.0));
    app.saveUiSettingsImmediatelyForTesting();

    QFile file(overridePath);
    QVERIFY(file.open(QIODevice::ReadOnly | QIODevice::Text));
    const QJsonDocument updated = QJsonDocument::fromJson(file.readAll());
    QVERIFY(updated.isObject());
    const QJsonObject updatedRisk = updated.object().value(QStringLiteral("riskRefresh")).toObject();
    QVERIFY(updatedRisk.value(QStringLiteral("enabled")).toBool());
    QCOMPARE(updatedRisk.value(QStringLiteral("intervalSeconds")).toDouble(), 9.0);
}

QTEST_MAIN(ApplicationSettingsPersistenceTest)
#include "ApplicationSettingsPersistenceTest.moc"
