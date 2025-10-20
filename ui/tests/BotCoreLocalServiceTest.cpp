#include <QtTest/QtTest>

#include "grpc/BotCoreLocalService.hpp"
#include "grpc/TradingClient.hpp"

namespace {
QString locateRepoRoot()
{
    QDir dir(QCoreApplication::applicationDirPath());
    for (int depth = 0; depth < 12; ++depth) {
        if (dir.exists(QStringLiteral("bot_core")) && dir.exists(QStringLiteral("ui")))
            return dir.absolutePath();
        if (!dir.cdUp())
            break;
    }
    dir = QDir(QDir::currentPath());
    for (int depth = 0; depth < 12; ++depth) {
        if (dir.exists(QStringLiteral("bot_core")) && dir.exists(QStringLiteral("ui")))
            return dir.absolutePath();
        if (!dir.cdUp())
            break;
    }
    return QDir::currentPath();
}
} // namespace

class BotCoreLocalServiceTest : public QObject {
    Q_OBJECT

private slots:
    void startsAndServesHistory();
};

void BotCoreLocalServiceTest::startsAndServesHistory()
{
    BotCoreLocalService service;
    service.setRepoRoot(locateRepoRoot());
    if (!service.start()) {
        QSKIP(qPrintable(QStringLiteral("Pomijam – stub bot_core nie wystartował: %1")
                             .arg(service.lastError())));
    }

    TradingClient client;
    TradingClient::InstrumentConfig config;
    config.exchange = QStringLiteral("BINANCE");
    config.symbol = QStringLiteral("BTC/USDT");
    config.venueSymbol = QStringLiteral("BTCUSDT");
    config.quoteCurrency = QStringLiteral("USDT");
    config.baseCurrency = QStringLiteral("BTC");
    config.granularityIso8601 = QStringLiteral("PT1M");

    client.setEndpoint(service.endpoint());
    client.setInstrument(config);
    client.setHistoryLimit(3);

    QSignalSpy historySpy(&client, &TradingClient::historyReceived);
    client.start();

    QTRY_VERIFY_WITH_TIMEOUT(historySpy.count() > 0, 5000);
    client.stop();
    service.stop();
}

QTEST_MAIN(BotCoreLocalServiceTest)
#include "BotCoreLocalServiceTest.moc"
