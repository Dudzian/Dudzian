#include <QtTest/QtTest>
#include <QCommandLineParser>
#include <QQmlApplicationEngine>
#include <QTemporaryDir>
#include <QFile>

#include "app/Application.hpp"
#include "grpc/TradingClient.hpp"

class ApplicationTransportValidationTest : public QObject {
    Q_OBJECT

private slots:
    void testInProcessModeRequiresDataset();
    void testInProcessModeAcceptsDataset();
    void testGrpcModeRejectsMissingRootCert();
    void testGrpcModeAcceptsValidTls();
};

void ApplicationTransportValidationTest::testInProcessModeRequiresDataset()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{QStringLiteral("test"),
                           QStringLiteral("--transport-mode"), QStringLiteral("in-process"),
                           QStringLiteral("--disable-metrics")};
    parser.process(args);

    QVERIFY(!app.applyParser(parser));
}

void ApplicationTransportValidationTest::testInProcessModeAcceptsDataset()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{QStringLiteral("test"),
                           QStringLiteral("--transport-mode"), QStringLiteral("in-process"),
                           QStringLiteral("--transport-dataset"), QStringLiteral("data/sample_ohlcv/trend.csv"),
                           QStringLiteral("--disable-metrics")};
    parser.process(args);

    QVERIFY(app.applyParser(parser));
    auto* client = app.tradingClientForTesting();
    QVERIFY(client);
    QCOMPARE(client->transportMode(), TradingClient::TransportMode::InProcess);
}

void ApplicationTransportValidationTest::testGrpcModeRejectsMissingRootCert()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString clientCert = dir.filePath(QStringLiteral("client.pem"));
    const QString clientKey = dir.filePath(QStringLiteral("client.key"));
    QFile certFile(clientCert);
    QVERIFY(certFile.open(QIODevice::WriteOnly));
    certFile.write("dummy");
    certFile.close();
    QFile keyFile(clientKey);
    QVERIFY(keyFile.open(QIODevice::WriteOnly));
    keyFile.write("dummy");
    keyFile.close();

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{QStringLiteral("test"),
                           QStringLiteral("--endpoint"), QStringLiteral("127.0.0.1:50061"),
                           QStringLiteral("--grpc-use-mtls"),
                           QStringLiteral("--grpc-client-cert"), clientCert,
                           QStringLiteral("--grpc-client-key"), clientKey,
                           QStringLiteral("--disable-metrics")};
    parser.process(args);

    QVERIFY(!app.applyParser(parser));
}

void ApplicationTransportValidationTest::testGrpcModeAcceptsValidTls()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString rootPath = dir.filePath(QStringLiteral("ca.pem"));
    const QString clientCert = dir.filePath(QStringLiteral("client.pem"));
    const QString clientKey = dir.filePath(QStringLiteral("client.key"));

    QFile rootFile(rootPath);
    QVERIFY(rootFile.open(QIODevice::WriteOnly));
    rootFile.write("dummy");
    rootFile.close();
    QFile certFile(clientCert);
    QVERIFY(certFile.open(QIODevice::WriteOnly));
    certFile.write("dummy");
    certFile.close();
    QFile keyFile(clientKey);
    QVERIFY(keyFile.open(QIODevice::WriteOnly));
    keyFile.write("dummy");
    keyFile.close();

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{QStringLiteral("test"),
                           QStringLiteral("--endpoint"), QStringLiteral("127.0.0.1:50061"),
                           QStringLiteral("--grpc-use-mtls"),
                           QStringLiteral("--grpc-root-cert"), rootPath,
                           QStringLiteral("--grpc-client-cert"), clientCert,
                           QStringLiteral("--grpc-client-key"), clientKey,
                           QStringLiteral("--disable-metrics")};
    parser.process(args);

    QVERIFY(app.applyParser(parser));
    auto* client = app.tradingClientForTesting();
    QVERIFY(client);
    QCOMPARE(client->transportMode(), TradingClient::TransportMode::Grpc);
}

QTEST_MAIN(ApplicationTransportValidationTest)
#include "ApplicationTransportValidationTest.moc"
