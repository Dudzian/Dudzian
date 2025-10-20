#include <QtTest/QtTest>
#include <QQmlApplicationEngine>
#include <QCommandLineParser>
#include <QDateTime>
#include <initializer_list>
#include <cstdlib>
#include <utility>

#include "app/Application.hpp"

namespace {

QStringList makeArgs(std::initializer_list<QString> options)
{
    QStringList args;
    args << QStringLiteral("test-app");
    args.append(options);
    return args;
}

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

class ApplicationRiskRefreshTest : public QObject {
    Q_OBJECT

private slots:
    void testDefaultConfiguration();
    void testCliOverride();
    void testCliDisable();
    void testEnvironmentOverrides();
    void testUiUpdateCycle();
    void testManualRefresh();
    void testScheduleSnapshot();
};

void ApplicationRiskRefreshTest::testDefaultConfiguration()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);
    parser.process(makeArgs({}));
    QVERIFY(app.applyParser(parser));

    QCOMPARE(app.riskRefreshIntervalMsForTesting(), 5000);
    QVERIFY(app.riskRefreshEnabledForTesting());

    app.startRiskRefreshTimerForTesting();
    QVERIFY(app.isRiskRefreshTimerActiveForTesting());
}

void ApplicationRiskRefreshTest::testCliOverride()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);
    parser.process(makeArgs({QStringLiteral("--risk-refresh-interval"), QStringLiteral("2.5")}));
    QVERIFY(app.applyParser(parser));

    QCOMPARE(app.riskRefreshIntervalMsForTesting(), 2500);
}

void ApplicationRiskRefreshTest::testCliDisable()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);
    parser.process(makeArgs({QStringLiteral("--risk-refresh-disable")}));
    QVERIFY(app.applyParser(parser));

    QVERIFY(!app.riskRefreshEnabledForTesting());
    app.startRiskRefreshTimerForTesting();
    QVERIFY(!app.isRiskRefreshTimerActiveForTesting());
}

void ApplicationRiskRefreshTest::testEnvironmentOverrides()
{
    EnvRestore intervalGuard(QByteArrayLiteral("BOT_CORE_UI_RISK_REFRESH_SECONDS"));
    EnvRestore disableGuard(QByteArrayLiteral("BOT_CORE_UI_RISK_REFRESH_DISABLE"));

    qputenv("BOT_CORE_UI_RISK_REFRESH_SECONDS", QByteArrayLiteral("3.5"));
    qunsetenv("BOT_CORE_UI_RISK_REFRESH_DISABLE");

    {
        QQmlApplicationEngine engine;
        Application app(engine);
        QCommandLineParser parser;
        app.configureParser(parser);
        parser.process(makeArgs({}));
        QVERIFY(app.applyParser(parser));
        QCOMPARE(app.riskRefreshIntervalMsForTesting(), 3500);
        QVERIFY(app.riskRefreshEnabledForTesting());
    }

    qputenv("BOT_CORE_UI_RISK_REFRESH_DISABLE", QByteArrayLiteral("true"));

    {
        QQmlApplicationEngine engine;
        Application app(engine);
        QCommandLineParser parser;
        app.configureParser(parser);
        parser.process(makeArgs({}));
        QVERIFY(app.applyParser(parser));
        QVERIFY(!app.riskRefreshEnabledForTesting());
    }
}

void ApplicationRiskRefreshTest::testUiUpdateCycle()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto snapshot = app.riskRefreshSnapshot();
    QVERIFY(snapshot.value(QStringLiteral("enabled")).toBool());
    QCOMPARE(snapshot.value(QStringLiteral("intervalSeconds")).toDouble(), 5.0);
    QVERIFY(!snapshot.value(QStringLiteral("active")).toBool());
    QVERIFY(snapshot.value(QStringLiteral("nextRefreshDueAt")).toString().isEmpty());

    QVERIFY(app.updateRiskRefresh(true, 3.2));
    QCOMPARE(app.riskRefreshIntervalMsForTesting(), 3200);
    snapshot = app.riskRefreshSnapshot();
    QCOMPARE(snapshot.value(QStringLiteral("intervalSeconds")).toDouble(), 3.2);

    app.startRiskRefreshTimerForTesting();
    QVERIFY(app.isRiskRefreshTimerActiveForTesting());
    snapshot = app.riskRefreshSnapshot();
    QVERIFY(snapshot.value(QStringLiteral("active")).toBool());
    QVERIFY(!snapshot.value(QStringLiteral("nextRefreshDueAt")).toString().isEmpty());

    QVERIFY(app.updateRiskRefresh(true, 4.0));
    QCOMPARE(app.riskRefreshIntervalMsForTesting(), 4000);
    QVERIFY(app.isRiskRefreshTimerActiveForTesting());

    QVERIFY(app.updateRiskRefresh(false, 0.0));
    QVERIFY(!app.riskRefreshEnabledForTesting());
    QVERIFY(!app.isRiskRefreshTimerActiveForTesting());

    QVERIFY(!app.updateRiskRefresh(true, 0.0));
    QVERIFY(!app.riskRefreshEnabledForTesting());

    snapshot = app.riskRefreshSnapshot();
    QVERIFY(!snapshot.value(QStringLiteral("enabled")).toBool());
    QCOMPARE(snapshot.value(QStringLiteral("intervalSeconds")).toDouble(), 4.0);
    QVERIFY(snapshot.value(QStringLiteral("nextRefreshDueAt")).toString().isEmpty());
    QCOMPARE(snapshot.value(QStringLiteral("nextRefreshInSeconds")).toDouble(), -1.0);
}

void ApplicationRiskRefreshTest::testManualRefresh()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);
    parser.process(makeArgs({QStringLiteral("--risk-refresh-interval"), QStringLiteral("2.5")}));
    QVERIFY(app.applyParser(parser));

    app.startRiskRefreshTimerForTesting();
    const auto initialNext = app.nextRiskRefreshDueUtcForTesting();
    QVERIFY(initialNext.isValid());

    QVERIFY(app.triggerRiskRefreshNow());
    const auto requestAt = app.lastRiskRefreshRequestUtcForTesting();
    QVERIFY(requestAt.isValid());
    const auto nextAfterManual = app.nextRiskRefreshDueUtcForTesting();
    QVERIFY(nextAfterManual.isValid());
    QVERIFY(nextAfterManual >= initialNext);

    auto snapshot = app.riskRefreshSnapshot();
    QVERIFY(!snapshot.value(QStringLiteral("lastRequestAt")).toString().isEmpty());
    QVERIFY(snapshot.value(QStringLiteral("nextRefreshInSeconds")).toDouble() >= 0.0);
}

void ApplicationRiskRefreshTest::testScheduleSnapshot()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);
    parser.process(makeArgs({QStringLiteral("--risk-refresh-interval"), QStringLiteral("1.5")}));
    QVERIFY(app.applyParser(parser));

    QVariantMap snapshot = app.riskRefreshSnapshot();
    QVERIFY(snapshot.value(QStringLiteral("nextRefreshDueAt")).toString().isEmpty());
    QCOMPARE(snapshot.value(QStringLiteral("nextRefreshInSeconds")).toDouble(), -1.0);

    app.startRiskRefreshTimerForTesting();
    snapshot = app.riskRefreshSnapshot();
    QVERIFY(snapshot.value(QStringLiteral("active")).toBool());
    QVERIFY(snapshot.value(QStringLiteral("nextRefreshInSeconds")).toDouble() >= 0.0);

    const QDateTime updateTime = QDateTime::currentDateTimeUtc();
    RiskSnapshotData riskSnapshot;
    riskSnapshot.generatedAt = updateTime;
    riskSnapshot.hasData = true;
    QMetaObject::invokeMethod(&app, "handleRiskState", Q_ARG(RiskSnapshotData, riskSnapshot));
    snapshot = app.riskRefreshSnapshot();
    QCOMPARE(snapshot.value(QStringLiteral("lastUpdateAt")).toString(), updateTime.toString(Qt::ISODateWithMs));

    QVERIFY(app.updateRiskRefresh(false, 1.5));
    snapshot = app.riskRefreshSnapshot();
    QVERIFY(!snapshot.value(QStringLiteral("active")).toBool());
    QVERIFY(snapshot.value(QStringLiteral("nextRefreshDueAt")).toString().isEmpty());
    QCOMPARE(snapshot.value(QStringLiteral("nextRefreshInSeconds")).toDouble(), -1.0);
}

QTEST_MAIN(ApplicationRiskRefreshTest)
#include "ApplicationRiskRefreshTest.moc"
