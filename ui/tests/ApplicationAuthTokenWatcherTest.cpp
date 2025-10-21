#include <QtTest/QtTest>

#include <QQmlApplicationEngine>
#include <QCommandLineParser>
#include <QDir>
#include <QFile>
#include <QTemporaryDir>
#include <QTextStream>

#include <memory>

#include "app/Application.hpp"
#include "health/HealthStatusController.hpp"
#include "grpc/HealthClient.hpp"
#include "telemetry/TelemetryReporter.hpp"

namespace {

void writeTextFile(const QString& path, const QString& value)
{
    QFile file(path);
    QVERIFY2(file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text),
             "Nie można zapisać pliku");
    QTextStream stream(&file);
    stream << value;
    file.close();
}

void writeTokenFile(const QString& path, const QString& value)
{
    writeTextFile(path, value);
}

bool metadataHasAuthorization(TradingClient* client, const QByteArray& expected)
{
    const auto metadata = client->authMetadataForTesting();
    for (const auto& entry : metadata) {
        if (entry.first == QByteArrayLiteral("authorization")) {
            return entry.second == expected;
        }
    }
    return false;
}

bool metadataHasAuthorizationHeader(TradingClient* client)
{
    const auto metadata = client->authMetadataForTesting();
    for (const auto& entry : metadata) {
        if (entry.first == QByteArrayLiteral("authorization"))
            return true;
    }
    return false;
}

class RecordingTelemetryReporter final : public TelemetryReporter {
public:
    void setEnabled(bool enabled) override { m_enabled = enabled; }
    void setEndpoint(const QString& endpoint) override { m_endpoint = endpoint; }
    void setNotesTag(const QString& tag) override { m_tag = tag; }
    void setWindowCount(int count) override { m_windowCount = count; }
    void setTlsConfig(const TelemetryTlsConfig& config) override
    {
        ++tlsConfigCallCount;
        m_tlsConfig = config;
    }
    void setAuthToken(const QString& token) override { authToken = token; }
    void setRbacRole(const QString& role) override { m_role = role; }
    void setScreenInfo(const ScreenInfo& info) override { m_screenInfo = info; }
    void clearScreenInfo() override { m_screenInfo.reset(); }
    bool isEnabled() const override { return m_enabled; }

    void reportReduceMotion(const PerformanceGuard&, bool, double, int, int) override { }
    void reportOverlayBudget(const PerformanceGuard&, int, int, bool) override { }
    void reportJankEvent(const PerformanceGuard&, double, double, bool, int, int) override { }

    QString authToken;
    int     tlsConfigCallCount = 0;

private:
    bool m_enabled = false;
    QString m_endpoint;
    QString m_tag;
    int m_windowCount = 0;
    TelemetryTlsConfig m_tlsConfig;
    QString m_role;
    std::optional<ScreenInfo> m_screenInfo;
};

class RecordingHealthClient : public HealthClientInterface {
public:
    void setEndpoint(const QString& endpoint) override { endpointSeen = endpoint; }
    void setTlsConfig(const GrpcTlsConfig& config) override
    {
        tlsConfig = config;
        ++tlsConfigCalls;
    }
    void setAuthToken(const QString& token) override { authToken = token; }
    void setRbacRole(const QString& role) override { rbacRole = role; }
    void setRbacScopes(const QStringList& scopes) override { rbacScopes = scopes; }
    QVector<QPair<QByteArray, QByteArray>> authMetadataForTesting() const override { return {}; }
    HealthCheckResult check() override { return {}; }

    QString endpointSeen;
    GrpcTlsConfig tlsConfig;
    QString authToken;
    QString rbacRole;
    QStringList rbacScopes;
    int tlsConfigCalls = 0;
};

} // namespace

class ApplicationAuthTokenWatcherTest : public QObject {
    Q_OBJECT

private slots:
    void tradingTokenFileReloadsAuthMetadata();
    void metricsTokenFileReloadsTelemetryReporter();
    void healthTokenFileReloadsAndFallsBackToTradingToken();
    void tradingTlsWatcherReloadsClientOnCertificateChange();
    void metricsTlsWatcherReloadsReporterOnTlsMaterialUpdate();
    void healthTlsWatcherReloadsClientOnKeyRotation();
    void tradingTokenWatcherHandlesDeferredCreation();
    void metricsTlsWatcherHandlesDeferredDirectoryCreation();
};

void ApplicationAuthTokenWatcherTest::tradingTokenFileReloadsAuthMetadata()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString tokenPath = dir.filePath(QStringLiteral("trading.token"));
    writeTokenFile(tokenPath, QStringLiteral("first-token"));

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--grpc-auth-token-file"), tokenPath,
        QStringLiteral("--metrics-endpoint"), QStringLiteral("localhost:9000")
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    TradingClient* client = app.tradingClientForTesting();
    QVERIFY(client != nullptr);

    QTRY_VERIFY_WITH_TIMEOUT(metadataHasAuthorization(client, QByteArrayLiteral("Bearer first-token")), 1000);

    writeTokenFile(tokenPath, QStringLiteral("second-token"));
    QTRY_VERIFY_WITH_TIMEOUT(metadataHasAuthorization(client, QByteArrayLiteral("Bearer second-token")), 2000);

    writeTokenFile(tokenPath, QString());
    QTRY_VERIFY_WITH_TIMEOUT(!metadataHasAuthorizationHeader(client), 2000);
}

void ApplicationAuthTokenWatcherTest::metricsTokenFileReloadsTelemetryReporter()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto reporter = std::make_unique<RecordingTelemetryReporter>();
    RecordingTelemetryReporter* reporterPtr = reporter.get();
    app.setTelemetryReporter(std::move(reporter));

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString tokenPath = dir.filePath(QStringLiteral("metrics.token"));
    writeTokenFile(tokenPath, QStringLiteral("alpha"));

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--metrics-endpoint"), QStringLiteral("metrics:7000"),
        QStringLiteral("--metrics-auth-token-file"), tokenPath
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    QTRY_COMPARE_WITH_TIMEOUT(reporterPtr->authToken, QStringLiteral("alpha"), 1000);

    writeTokenFile(tokenPath, QStringLiteral("beta"));
    QTRY_COMPARE_WITH_TIMEOUT(reporterPtr->authToken, QStringLiteral("beta"), 2000);

    writeTokenFile(tokenPath, QString());
    QTRY_COMPARE_WITH_TIMEOUT(reporterPtr->authToken, QString(), 2000);
}

void ApplicationAuthTokenWatcherTest::healthTokenFileReloadsAndFallsBackToTradingToken()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto* controller = qobject_cast<HealthStatusController*>(app.healthController());
    QVERIFY(controller != nullptr);

    auto client = std::make_shared<RecordingHealthClient>();
    controller->setHealthClientForTesting(client);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString tokenPath = dir.filePath(QStringLiteral("health.token"));
    writeTokenFile(tokenPath, QStringLiteral("health-one"));

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("trading:50051"),
        QStringLiteral("--grpc-auth-token"), QStringLiteral("trading-master"),
        QStringLiteral("--health-endpoint"), QStringLiteral("health:50052"),
        QStringLiteral("--health-auth-token-file"), tokenPath
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    QTRY_COMPARE_WITH_TIMEOUT(client->authToken, QStringLiteral("health-one"), 1000);

    writeTokenFile(tokenPath, QStringLiteral("health-two"));
    QTRY_COMPARE_WITH_TIMEOUT(client->authToken, QStringLiteral("health-two"), 2000);

    writeTokenFile(tokenPath, QString());
    QTRY_COMPARE_WITH_TIMEOUT(client->authToken, QStringLiteral("trading-master"), 2000);
}

void ApplicationAuthTokenWatcherTest::tradingTlsWatcherReloadsClientOnCertificateChange()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString rootPath = dir.filePath(QStringLiteral("ca.pem"));
    const QString certPath = dir.filePath(QStringLiteral("client.pem"));
    const QString keyPath = dir.filePath(QStringLiteral("client.key"));
    writeTextFile(rootPath, QStringLiteral("root-one"));
    writeTextFile(certPath, QStringLiteral("cert-one"));
    writeTextFile(keyPath, QStringLiteral("key-one"));

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--grpc-use-mtls"),
        QStringLiteral("--grpc-root-cert"), rootPath,
        QStringLiteral("--grpc-client-cert"), certPath,
        QStringLiteral("--grpc-client-key"), keyPath,
        QStringLiteral("--metrics-endpoint"), QStringLiteral("localhost:9000"),
        QStringLiteral("--disable-metrics")
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    const quint64 initialGeneration = app.tradingTlsReloadGenerationForTesting();
    QVERIFY(initialGeneration > 0);

    writeTextFile(rootPath, QStringLiteral("root-two"));
    QTRY_VERIFY_WITH_TIMEOUT(app.tradingTlsReloadGenerationForTesting() > initialGeneration, 2000);
    const quint64 afterRootUpdate = app.tradingTlsReloadGenerationForTesting();

    writeTextFile(keyPath, QStringLiteral("key-two"));
    QTRY_VERIFY_WITH_TIMEOUT(app.tradingTlsReloadGenerationForTesting() > afterRootUpdate, 2000);
}

void ApplicationAuthTokenWatcherTest::metricsTlsWatcherReloadsReporterOnTlsMaterialUpdate()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto reporter = std::make_unique<RecordingTelemetryReporter>();
    RecordingTelemetryReporter* reporterPtr = reporter.get();
    app.setTelemetryReporter(std::move(reporter));

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString rootPath = dir.filePath(QStringLiteral("metrics-ca.pem"));
    const QString certPath = dir.filePath(QStringLiteral("metrics-client.pem"));
    const QString keyPath = dir.filePath(QStringLiteral("metrics-client.key"));
    writeTextFile(rootPath, QStringLiteral("root-a"));
    writeTextFile(certPath, QStringLiteral("cert-a"));
    writeTextFile(keyPath, QStringLiteral("key-a"));

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--metrics-endpoint"), QStringLiteral("metrics:7000"),
        QStringLiteral("--metrics-use-tls"),
        QStringLiteral("--metrics-root-cert"), rootPath,
        QStringLiteral("--metrics-client-cert"), certPath,
        QStringLiteral("--metrics-client-key"), keyPath
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    QTRY_VERIFY_WITH_TIMEOUT(reporterPtr->tlsConfigCallCount > 0, 2000);
    int lastCount = reporterPtr->tlsConfigCallCount;

    writeTextFile(certPath, QStringLiteral("cert-b"));
    QTRY_VERIFY_WITH_TIMEOUT(reporterPtr->tlsConfigCallCount > lastCount, 2000);
    lastCount = reporterPtr->tlsConfigCallCount;

    writeTextFile(keyPath, QStringLiteral("key-b"));
    QTRY_VERIFY_WITH_TIMEOUT(reporterPtr->tlsConfigCallCount > lastCount, 2000);
}

void ApplicationAuthTokenWatcherTest::healthTlsWatcherReloadsClientOnKeyRotation()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto* controller = qobject_cast<HealthStatusController*>(app.healthController());
    QVERIFY(controller != nullptr);

    auto client = std::make_shared<RecordingHealthClient>();
    controller->setHealthClientForTesting(client);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString rootPath = dir.filePath(QStringLiteral("health-ca.pem"));
    const QString certPath = dir.filePath(QStringLiteral("health-client.pem"));
    const QString keyPath = dir.filePath(QStringLiteral("health-client.key"));
    writeTextFile(rootPath, QStringLiteral("root-x"));
    writeTextFile(certPath, QStringLiteral("cert-x"));
    writeTextFile(keyPath, QStringLiteral("key-x"));

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("trading:50051"),
        QStringLiteral("--grpc-auth-token"), QStringLiteral("trading-master"),
        QStringLiteral("--health-endpoint"), QStringLiteral("health:50052"),
        QStringLiteral("--health-use-tls"),
        QStringLiteral("--health-tls-root-cert"), rootPath,
        QStringLiteral("--health-tls-client-cert"), certPath,
        QStringLiteral("--health-tls-client-key"), keyPath
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    QTRY_VERIFY_WITH_TIMEOUT(client->tlsConfigCalls > 0, 2000);
    int lastCount = client->tlsConfigCalls;

    writeTextFile(certPath, QStringLiteral("cert-y"));
    QTRY_VERIFY_WITH_TIMEOUT(client->tlsConfigCalls > lastCount, 2000);
    lastCount = client->tlsConfigCalls;

    writeTextFile(keyPath, QStringLiteral("key-y"));
    QTRY_VERIFY_WITH_TIMEOUT(client->tlsConfigCalls > lastCount, 2000);
}

void ApplicationAuthTokenWatcherTest::tradingTokenWatcherHandlesDeferredCreation()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    const QString pendingDir = QDir(dir.path()).filePath(QStringLiteral("tokens/pending"));
    const QString tokenPath = QDir(pendingDir).filePath(QStringLiteral("trading.token"));
    QVERIFY(!QDir(pendingDir).exists());

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--grpc-auth-token-file"), tokenPath,
        QStringLiteral("--metrics-endpoint"), QStringLiteral("metrics:9000")
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    TradingClient* client = app.tradingClientForTesting();
    QVERIFY(client != nullptr);
    QVERIFY(!metadataHasAuthorizationHeader(client));

    QVERIFY(QDir().mkpath(pendingDir));
    writeTokenFile(tokenPath, QStringLiteral("deferred-token"));

    QTRY_VERIFY_WITH_TIMEOUT(metadataHasAuthorization(client, QByteArrayLiteral("Bearer deferred-token")), 3000);
}

void ApplicationAuthTokenWatcherTest::metricsTlsWatcherHandlesDeferredDirectoryCreation()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto reporter = std::make_unique<RecordingTelemetryReporter>();
    RecordingTelemetryReporter* reporterPtr = reporter.get();
    app.setTelemetryReporter(std::move(reporter));

    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    const QString tlsDir = QDir(dir.path()).filePath(QStringLiteral("tls/pending"));
    const QString rootPath = QDir(tlsDir).filePath(QStringLiteral("ca.pem"));
    const QString certPath = QDir(tlsDir).filePath(QStringLiteral("client.pem"));
    const QString keyPath = QDir(tlsDir).filePath(QStringLiteral("client.key"));
    QVERIFY(!QDir(tlsDir).exists());

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--metrics-endpoint"), QStringLiteral("metrics:9000"),
        QStringLiteral("--metrics-use-tls"),
        QStringLiteral("--metrics-root-cert"), rootPath,
        QStringLiteral("--metrics-client-cert"), certPath,
        QStringLiteral("--metrics-client-key"), keyPath
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    const int initialCalls = reporterPtr->tlsConfigCallCount;

    QVERIFY(QDir().mkpath(tlsDir));
    writeTextFile(rootPath, QStringLiteral("root"));
    writeTextFile(certPath, QStringLiteral("cert"));
    writeTextFile(keyPath, QStringLiteral("key"));

    QTRY_VERIFY_WITH_TIMEOUT(reporterPtr->tlsConfigCallCount > initialCalls, 3000);
    QCOMPARE(reporterPtr->tlsConfig.rootCertificatePath, rootPath);
    QCOMPARE(reporterPtr->tlsConfig.clientCertificatePath, certPath);
    QCOMPARE(reporterPtr->tlsConfig.clientKeyPath, keyPath);
}

QTEST_MAIN(ApplicationAuthTokenWatcherTest)
#include "ApplicationAuthTokenWatcherTest.moc"
