#include <QtTest/QtTest>
#include <QQmlApplicationEngine>
#include <QScopeGuard>

#include <memory>

#include "app/Application.hpp"
#include "health/HealthStatusController.hpp"
#include "grpc/HealthClient.hpp"

class RecordingHealthClient : public HealthClientInterface {
public:
    void setEndpoint(const QString& endpoint) override { endpointSeen = endpoint; }
    void setTlsConfig(const GrpcTlsConfig& config) override {
        tlsConfigured = true;
        tlsConfig = config;
    }
    void setAuthToken(const QString& token) override { authToken = token; }
    void setRbacRole(const QString& role) override { rbacRole = role; }
    void setRbacScopes(const QStringList& scopes) override { rbacScopes = scopes; }

    QVector<QPair<QByteArray, QByteArray>> authMetadataForTesting() const override { return {}; }
    HealthCheckResult check() override { return {}; }

    QString                 endpointSeen;
    bool                    tlsConfigured = false;
    GrpcTlsConfig           tlsConfig;
    QString                 authToken;
    QString                 rbacRole;
    QStringList             rbacScopes;
};

class ApplicationHealthConfigTest final : public QObject {
    Q_OBJECT

private slots:
    void testHealthTlsInheritsTrading();
    void testHealthTlsCliOverrides();
    void testHealthTlsEnvironmentOverrides();
};

void ApplicationHealthConfigTest::testHealthTlsInheritsTrading() {
    QQmlApplicationEngine engine;
    Application app(engine);

    auto* controller = qobject_cast<HealthStatusController*>(app.healthController());
    QVERIFY(controller != nullptr);
    auto client = std::make_shared<RecordingHealthClient>();
    controller->setHealthClientForTesting(client);

    QCommandLineParser parser;
    app.configureParser(parser);

    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"),
        QStringLiteral("trading:50051"),
        QStringLiteral("--health-endpoint"),
        QStringLiteral("health:50052"),
        QStringLiteral("--grpc-use-mtls"),
        QStringLiteral("--grpc-root-cert"),
        QStringLiteral("/tmp/trading_root.pem"),
        QStringLiteral("--grpc-client-cert"),
        QStringLiteral("/tmp/trading_client.pem"),
        QStringLiteral("--grpc-client-key"),
        QStringLiteral("/tmp/trading_client.key"),
        QStringLiteral("--grpc-target-name"),
        QStringLiteral("trading.internal")
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    QVERIFY(client->tlsConfigured);
    QCOMPARE(client->tlsConfig.enabled, true);
    QCOMPARE(client->tlsConfig.rootCertificatePath, QStringLiteral("/tmp/trading_root.pem"));
    QCOMPARE(client->tlsConfig.clientCertificatePath, QStringLiteral("/tmp/trading_client.pem"));
    QCOMPARE(client->tlsConfig.clientKeyPath, QStringLiteral("/tmp/trading_client.key"));
    QCOMPARE(client->tlsConfig.targetNameOverride, QStringLiteral("trading.internal"));
}

void ApplicationHealthConfigTest::testHealthTlsCliOverrides() {
    QQmlApplicationEngine engine;
    Application app(engine);

    auto* controller = qobject_cast<HealthStatusController*>(app.healthController());
    QVERIFY(controller != nullptr);
    auto client = std::make_shared<RecordingHealthClient>();
    controller->setHealthClientForTesting(client);

    QCommandLineParser parser;
    app.configureParser(parser);

    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"),
        QStringLiteral("trading:50051"),
        QStringLiteral("--health-endpoint"),
        QStringLiteral("health:50052"),
        QStringLiteral("--grpc-use-mtls"),
        QStringLiteral("--grpc-root-cert"),
        QStringLiteral("/tmp/trading_root.pem"),
        QStringLiteral("--health-use-tls"),
        QStringLiteral("--health-tls-root-cert"),
        QStringLiteral("/etc/health/root.pem"),
        QStringLiteral("--health-tls-client-cert"),
        QStringLiteral("/etc/health/client.pem"),
        QStringLiteral("--health-tls-client-key"),
        QStringLiteral("/etc/health/client.key"),
        QStringLiteral("--health-tls-server-name"),
        QStringLiteral("health.internal"),
        QStringLiteral("--health-tls-target-name"),
        QStringLiteral("health-target"),
        QStringLiteral("--health-tls-pinned-sha256"),
        QStringLiteral("cafebabe"),
        QStringLiteral("--health-tls-require-client-auth")
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    QVERIFY(client->tlsConfigured);
    QCOMPARE(client->tlsConfig.enabled, true);
    QCOMPARE(client->tlsConfig.rootCertificatePath, QStringLiteral("/etc/health/root.pem"));
    QCOMPARE(client->tlsConfig.clientCertificatePath, QStringLiteral("/etc/health/client.pem"));
    QCOMPARE(client->tlsConfig.clientKeyPath, QStringLiteral("/etc/health/client.key"));
    QCOMPARE(client->tlsConfig.serverNameOverride, QStringLiteral("health.internal"));
    QCOMPARE(client->tlsConfig.targetNameOverride, QStringLiteral("health-target"));
    QCOMPARE(client->tlsConfig.pinnedServerFingerprint, QStringLiteral("cafebabe"));
    QCOMPARE(client->tlsConfig.requireClientAuth, true);
}

void ApplicationHealthConfigTest::testHealthTlsEnvironmentOverrides() {
    QQmlApplicationEngine engine;
    Application app(engine);

    auto* controller = qobject_cast<HealthStatusController*>(app.healthController());
    QVERIFY(controller != nullptr);
    auto client = std::make_shared<RecordingHealthClient>();
    controller->setHealthClientForTesting(client);

    auto restoreEnv = qScopeGuard([]() {
        qunsetenv("BOT_CORE_UI_HEALTH_USE_TLS");
        qunsetenv("BOT_CORE_UI_HEALTH_TLS_REQUIRE_CLIENT_AUTH");
        qunsetenv("BOT_CORE_UI_HEALTH_TLS_ROOT_CERT");
        qunsetenv("BOT_CORE_UI_HEALTH_TLS_CLIENT_CERT");
        qunsetenv("BOT_CORE_UI_HEALTH_TLS_CLIENT_KEY");
        qunsetenv("BOT_CORE_UI_HEALTH_TLS_SERVER_NAME");
        qunsetenv("BOT_CORE_UI_HEALTH_TLS_TARGET_NAME");
        qunsetenv("BOT_CORE_UI_HEALTH_TLS_PINNED_SHA256");
    });

    qputenv("BOT_CORE_UI_HEALTH_USE_TLS", QByteArrayLiteral("true"));
    qputenv("BOT_CORE_UI_HEALTH_TLS_REQUIRE_CLIENT_AUTH", QByteArrayLiteral("false"));
    qputenv("BOT_CORE_UI_HEALTH_TLS_ROOT_CERT", QByteArrayLiteral("/var/health/root.pem"));
    qputenv("BOT_CORE_UI_HEALTH_TLS_CLIENT_CERT", QByteArrayLiteral("/var/health/client.pem"));
    qputenv("BOT_CORE_UI_HEALTH_TLS_CLIENT_KEY", QByteArrayLiteral("/var/health/client.key"));
    qputenv("BOT_CORE_UI_HEALTH_TLS_SERVER_NAME", QByteArrayLiteral("health.prod"));
    qputenv("BOT_CORE_UI_HEALTH_TLS_TARGET_NAME", QByteArrayLiteral("prod-target"));
    qputenv("BOT_CORE_UI_HEALTH_TLS_PINNED_SHA256", QByteArrayLiteral("deadbeef"));

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"),
        QStringLiteral("trading:50051"),
        QStringLiteral("--health-endpoint"),
        QStringLiteral("health:50052"),
        QStringLiteral("--grpc-use-mtls"),
        QStringLiteral("--grpc-root-cert"),
        QStringLiteral("/tmp/trading_root.pem")
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    QVERIFY(client->tlsConfigured);
    QCOMPARE(client->tlsConfig.enabled, true);
    QCOMPARE(client->tlsConfig.requireClientAuth, false);
    QCOMPARE(client->tlsConfig.rootCertificatePath, QStringLiteral("/var/health/root.pem"));
    QCOMPARE(client->tlsConfig.clientCertificatePath, QStringLiteral("/var/health/client.pem"));
    QCOMPARE(client->tlsConfig.clientKeyPath, QStringLiteral("/var/health/client.key"));
    QCOMPARE(client->tlsConfig.serverNameOverride, QStringLiteral("health.prod"));
    QCOMPARE(client->tlsConfig.targetNameOverride, QStringLiteral("prod-target"));
    QCOMPARE(client->tlsConfig.pinnedServerFingerprint, QStringLiteral("deadbeef"));
}

QTEST_MAIN(ApplicationHealthConfigTest)
#include "ApplicationHealthConfigTest.moc"
