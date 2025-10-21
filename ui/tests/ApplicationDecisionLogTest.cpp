#include <QCommandLineParser>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QQmlApplicationEngine>
#include <QTemporaryDir>
#include <QtTest/QtTest>

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

QTEST_MAIN(ApplicationDecisionLogTest)
#include "ApplicationDecisionLogTest.moc"
