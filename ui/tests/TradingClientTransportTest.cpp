#include <QtTest/QtTest>
#include <QSignalSpy>

#include "grpc/TradingClient.hpp"

class TradingClientTransportTest : public QObject {
    Q_OBJECT

private slots:
    void testTransportSwitchingLifecycle();
    void testGrpcStartWithoutEndpoint();
    void testInProcessDatasetRestart();
};

void TradingClientTransportTest::testTransportSwitchingLifecycle()
{
    TradingClient client;
    client.setTransportMode(TradingClient::TransportMode::InProcess);
    client.setInProcessDatasetPath(QStringLiteral("data/sample_ohlcv/trend.csv"));

    QSignalSpy historySpy(&client, &TradingClient::historyReceived);
    client.start();
    QVERIFY(historySpy.wait(2000));
    QVERIFY(historySpy.count() > 0);
    client.stop();
    QVERIFY(!client.hasGrpcChannelForTesting());

    client.setTransportMode(TradingClient::TransportMode::Grpc);
    client.setEndpoint(QStringLiteral("127.0.0.1:65535"));

    QSignalSpy stateSpy(&client, &TradingClient::connectionStateChanged);
    client.start();
    QVERIFY(stateSpy.wait(1000));
    QTRY_VERIFY_WITH_TIMEOUT(client.hasGrpcChannelForTesting(), 1000);
    client.stop();
    QVERIFY(!client.hasGrpcChannelForTesting());
}

void TradingClientTransportTest::testGrpcStartWithoutEndpoint()
{
    TradingClient client;
    client.setEndpoint(QString());

    QSignalSpy stateSpy(&client, &TradingClient::connectionStateChanged);
    client.start();
    QVERIFY(stateSpy.wait(500));
    QVERIFY(!stateSpy.isEmpty());
    const auto lastState = stateSpy.takeLast().at(0).toString();
    QCOMPARE(lastState, QStringLiteral("unavailable"));
    client.stop();
}

void TradingClientTransportTest::testInProcessDatasetRestart()
{
    TradingClient client;
    client.setTransportMode(TradingClient::TransportMode::InProcess);
    client.setInProcessDatasetPath(QStringLiteral("data/sample_ohlcv/trend.csv"));

    QSignalSpy historySpy(&client, &TradingClient::historyReceived);
    client.start();
    QVERIFY(historySpy.wait(2000));
    const int initialCount = historySpy.count();

    client.setInProcessDatasetPath(QStringLiteral("data/sample_ohlcv/trend_missing.csv"));
    QVERIFY(historySpy.wait(2000));
    QVERIFY(historySpy.count() > initialCount);

    client.stop();
}

QTEST_MAIN(TradingClientTransportTest)
#include "TradingClientTransportTest.moc"
