#include <QCommandLineParser>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QQmlApplicationEngine>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QtTest/QtTest>

#include "app/Application.hpp"
#include "app/ActivationController.hpp"

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

QString normalizePath(const QString& path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

} // namespace

class ApplicationDecisionLogTest : public QObject {
    Q_OBJECT

private slots:
    void appliesCliOverrides();
    void appliesEnvironmentOverrides();
    void usesDefaultFallback();
    void loadsSecurityCacheFromFile();
    void controlsLicenseRefreshSchedule();
};

void ApplicationDecisionLogTest::appliesCliOverrides()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    const QString logFile = dir.filePath(QStringLiteral("custom.jsonl"));
    QFile file(logFile);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.close();

    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);
    parser.process(makeArgs({QStringLiteral("--decision-log"), logFile,
                             QStringLiteral("--decision-log-limit"), QStringLiteral("12")}));

    QVERIFY(app.applyParser(parser));
    QCOMPARE(app.decisionLogPathForTesting(), normalizePath(logFile));
    QCOMPARE(app.decisionLogModelForTesting()->maximumEntries(), 12);
}

void ApplicationDecisionLogTest::appliesEnvironmentOverrides()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    const QString envFile = dir.filePath(QStringLiteral("env.jsonl"));
    QFile file(envFile);
    QVERIFY(file.open(QIODevice::WriteOnly));
    file.close();

    EnvRestore pathGuard(QByteArrayLiteral("BOT_CORE_UI_DECISION_LOG"));
    EnvRestore limitGuard(QByteArrayLiteral("BOT_CORE_UI_DECISION_LOG_LIMIT"));

    qputenv("BOT_CORE_UI_DECISION_LOG", envFile.toUtf8());
    qputenv("BOT_CORE_UI_DECISION_LOG_LIMIT", QByteArrayLiteral("37"));

    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);
    parser.process(makeArgs({}));

    QVERIFY(app.applyParser(parser));
    QCOMPARE(app.decisionLogPathForTesting(), normalizePath(envFile));
    QCOMPARE(app.decisionLogModelForTesting()->maximumEntries(), 37);
}

void ApplicationDecisionLogTest::usesDefaultFallback()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);
    parser.process(makeArgs({}));

    QVERIFY(app.applyParser(parser));

    const QString path = app.decisionLogPathForTesting();
    QVERIFY(QFileInfo(path).isAbsolute());
    QVERIFY(path.endsWith(QStringLiteral("logs/decision_journal")));
    QCOMPARE(app.decisionLogModelForTesting()->maximumEntries(), 250);
}

void ApplicationDecisionLogTest::loadsSecurityCacheFromFile()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    const QString cachePath = dir.filePath(QStringLiteral("cache.json"));
    QFile cacheFile(cachePath);
    QVERIFY(cacheFile.open(QIODevice::WriteOnly | QIODevice::Text));

    QJsonObject fingerprintPayload;
    fingerprintPayload.insert(QStringLiteral("fingerprint"), QStringLiteral("DEVICE-XYZ"));
    QJsonObject fingerprint;
    fingerprint.insert(QStringLiteral("payload"), fingerprintPayload);

    QJsonObject licenseSummary;
    licenseSummary.insert(QStringLiteral("edition"), QStringLiteral("Enterprise"));

    QJsonObject historyEntry;
    historyEntry.insert(QStringLiteral("licenseId"), QStringLiteral("OEM-123"));

    QJsonObject root;
    root.insert(QStringLiteral("fingerprint"), fingerprint);
    root.insert(QStringLiteral("oemLicense"), licenseSummary);
    root.insert(QStringLiteral("licenseHistory"), QJsonArray{historyEntry});
    root.insert(QStringLiteral("lastRefreshIso"), QStringLiteral("2024-01-01T00:00:00.000Z"));
    root.insert(QStringLiteral("lastRequestIso"), QStringLiteral("2024-01-01T00:00:00.000Z"));
    root.insert(QStringLiteral("nextRefreshIso"), QStringLiteral("2024-01-02T00:00:00.000Z"));
    root.insert(QStringLiteral("lastError"), QStringLiteral("Timeout podczas walidacji"));

    cacheFile.write(QJsonDocument(root).toJson(QJsonDocument::Compact));
    cacheFile.close();

    EnvRestore cacheGuard(QByteArrayLiteral("BOT_CORE_UI_LICENSE_CACHE_PATH"));
    EnvRestore intervalGuard(QByteArrayLiteral("BOT_CORE_UI_LICENSE_REFRESH_INTERVAL"));
    qputenv("BOT_CORE_UI_LICENSE_CACHE_PATH", cachePath.toUtf8());
    qputenv("BOT_CORE_UI_LICENSE_REFRESH_INTERVAL", QByteArrayLiteral("0"));

    QQmlApplicationEngine engine;
    Application app(engine);

    const QVariantMap cache = app.securityCacheForTesting();
    const QVariantMap cachedFingerprint = cache.value(QStringLiteral("fingerprint")).toMap();
    QCOMPARE(cachedFingerprint.value(QStringLiteral("payload")).toMap().value(QStringLiteral("fingerprint")).toString(),
             QStringLiteral("DEVICE-XYZ"));
    QCOMPARE(cache.value(QStringLiteral("oemLicense")).toMap().value(QStringLiteral("edition")).toString(),
             QStringLiteral("Enterprise"));
    QCOMPARE(cache.value(QStringLiteral("lastError")).toString(), QStringLiteral("Timeout podczas walidacji"));

    auto activation = qobject_cast<ActivationController*>(app.activationController());
    QVERIFY(activation);
    QCOMPARE(activation->fingerprint().value(QStringLiteral("payload")).toMap().value(QStringLiteral("fingerprint")).toString(),
             QStringLiteral("DEVICE-XYZ"));
    QVERIFY(!activation->licenses().isEmpty());
    QCOMPARE(activation->licenses().first().toMap().value(QStringLiteral("licenseId")).toString(),
             QStringLiteral("OEM-123"));

    const QVariantMap schedule = app.licenseRefreshSchedule();
    QCOMPARE(schedule.value(QStringLiteral("lastCompletedAt")).toString(),
             QStringLiteral("2024-01-01T00:00:00.000Z"));
    QCOMPARE(schedule.value(QStringLiteral("nextRefreshDueAt")).toString(),
             QStringLiteral("2024-01-02T00:00:00.000Z"));
    QVERIFY(!schedule.value(QStringLiteral("active")).toBool());
}

void ApplicationDecisionLogTest::controlsLicenseRefreshSchedule()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QVERIFY(app.setLicenseRefreshEnabled(false));
    QVariantMap schedule = app.licenseRefreshSchedule();
    QVERIFY(!schedule.value(QStringLiteral("active")).toBool());

    QVERIFY(app.setLicenseRefreshIntervalSeconds(120));
    schedule = app.licenseRefreshSchedule();
    QCOMPARE(schedule.value(QStringLiteral("intervalSeconds")).toInt(), 120);

    QVERIFY(app.setLicenseRefreshEnabled(true));
    schedule = app.licenseRefreshSchedule();
    QVERIFY(schedule.value(QStringLiteral("active")).toBool());
    QCOMPARE(schedule.value(QStringLiteral("intervalSeconds")).toInt(), 120);

    QVERIFY(app.triggerLicenseRefreshNow());
    schedule = app.licenseRefreshSchedule();
    QVERIFY(!schedule.value(QStringLiteral("lastRequestAt")).toString().isEmpty());

    QVERIFY(app.setLicenseRefreshIntervalSeconds(0));
    schedule = app.licenseRefreshSchedule();
    QCOMPARE(schedule.value(QStringLiteral("intervalSeconds")).toInt(), 0);
    QVERIFY(!schedule.value(QStringLiteral("active")).toBool());

    const QVariantMap cache = app.securityCacheForTesting();
    QCOMPARE(cache.value(QStringLiteral("refreshIntervalSeconds")).toInt(), 0);
    QVERIFY(!cache.value(QStringLiteral("refreshActive")).toBool());
}

QTEST_MAIN(ApplicationDecisionLogTest)
#include "ApplicationDecisionLogTest.moc"
