#include "Application.hpp"

#include <QQmlContext>

Application::Application(QQmlApplicationEngine& engine, QObject* parent)
    : QObject(parent)
    , m_engine(engine) {
    exposeToQml();

    connect(&m_client, &TradingClient::historyReceived, this, &Application::handleHistory);
    connect(&m_client, &TradingClient::candleReceived, this, &Application::handleCandle);
    connect(&m_client, &TradingClient::connectionStateChanged, this, [this](const QString& status) {
        m_connectionStatus = status;
        Q_EMIT connectionStatusChanged();
    });
    connect(&m_client, &TradingClient::performanceGuardUpdated, this, [this](const PerformanceGuard& guard) {
        m_guard = guard;
        Q_EMIT performanceGuardChanged();
    });
    connect(&m_client, &TradingClient::streamingChanged, this, [this]() {
        const QString state = m_client.isStreaming() ? QStringLiteral("streaming") : QStringLiteral("idle");
        m_connectionStatus = state;
        Q_EMIT connectionStatusChanged();
    });
}

void Application::configureParser(QCommandLineParser& parser) const {
    parser.addHelpOption();
    parser.addOption({{"e", "endpoint"}, tr("Adres gRPC host:port"), tr("endpoint"), QStringLiteral("127.0.0.1:50061")});
    parser.addOption({"exchange", tr("Nazwa giełdy"), tr("exchange"), QStringLiteral("BINANCE")});
    parser.addOption({"symbol", tr("Symbol logiczny"), tr("symbol"), QStringLiteral("BTC/USDT")});
    parser.addOption({"venue-symbol", tr("Symbol na giełdzie"), tr("venue"), QStringLiteral("BTCUSDT")});
    parser.addOption({"quote", tr("Waluta kwotowana"), tr("quote"), QStringLiteral("USDT")});
    parser.addOption({"base", tr("Waluta bazowa"), tr("base"), QStringLiteral("BTC")});
    parser.addOption({"granularity", tr("ISO-8601 duration"), tr("granularity"), QStringLiteral("PT1M")});
    parser.addOption({"history-limit", tr("Limit pobieranej historii"), tr("limit"), QStringLiteral("500")});
    parser.addOption({"max-samples", tr("Maksymalna liczba świec w modelu"), tr("samples"), QStringLiteral("10240")});
    parser.addOption({"fps-target", tr("Docelowy FPS"), tr("fps"), QStringLiteral("60")});
    parser.addOption({"reduce-motion-after", tr("Czas (s) po którym ograniczamy animacje"), tr("seconds"), QStringLiteral("1.0")});
    parser.addOption({"jank-threshold-ms", tr("Budżet janku w ms"), tr("ms"), QStringLiteral("18.0")});
    parser.addOption({"max-overlay-count", tr("Limit nakładek na wykres"), tr("count"), QStringLiteral("3")});
}

bool Application::applyParser(const QCommandLineParser& parser) {
    TradingClient::InstrumentConfig instrument;
    instrument.exchange = parser.value("exchange");
    instrument.symbol = parser.value("symbol");
    instrument.venueSymbol = parser.value("venue-symbol");
    instrument.quoteCurrency = parser.value("quote");
    instrument.baseCurrency = parser.value("base");
    instrument.granularityIso8601 = parser.value("granularity");

    m_client.setEndpoint(parser.value("endpoint"));
    m_client.setInstrument(instrument);

    const int historyLimit = parser.value("history-limit").toInt();
    m_client.setHistoryLimit(historyLimit);

    m_maxSamples = parser.value("max-samples").toInt();
    if (m_maxSamples <= 0) {
        m_maxSamples = 10240;
    }

    PerformanceGuard guard;
    guard.fpsTarget = parser.value("fps-target").toInt();
    guard.reduceMotionAfterSeconds = parser.value("reduce-motion-after").toDouble();
    guard.jankThresholdMs = parser.value("jank-threshold-ms").toDouble();
    guard.maxOverlayCount = parser.value("max-overlay-count").toInt();
    m_client.setPerformanceGuard(guard);
    m_guard = guard;
    Q_EMIT performanceGuardChanged();

    return true;
}

void Application::start() {
    m_ohlcvModel.setMaximumSamples(m_maxSamples);
    m_client.start();
}

void Application::stop() {
    m_client.stop();
}

void Application::handleHistory(const QList<OhlcvPoint>& candles) {
    m_ohlcvModel.resetWithHistory(candles);
}

void Application::handleCandle(const OhlcvPoint& candle) {
    m_ohlcvModel.applyIncrement(candle);
}

void Application::exposeToQml() {
    m_engine.rootContext()->setContextProperty(QStringLiteral("appController"), this);
    m_engine.rootContext()->setContextProperty(QStringLiteral("ohlcvModel"), &m_ohlcvModel);
}
