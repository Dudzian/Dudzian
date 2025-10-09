#include "Application.hpp"

#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QCommandLineParser>
#include <QQuickWindow>
#include <QtGlobal>

#include "telemetry/UiTelemetryReporter.hpp"
#include "utils/FrameRateMonitor.hpp"

Application::Application(QQmlApplicationEngine& engine, QObject* parent)
    : QObject(parent)
    , m_engine(engine) {
    // Startowe ustawienia instrumentu z klienta (mogą być nadpisane przez CLI)
    m_instrument = m_client.instrumentConfig();

    exposeToQml();

    // Podłącz okno po utworzeniu (dla FrameRateMonitor)
    connect(&m_engine, &QQmlApplicationEngine::objectCreated, this,
            [this](QObject* object, const QUrl&) { attachWindow(object); });

    // Połączenia sygnałów klienta
    connect(&m_client, &TradingClient::historyReceived, this, &Application::handleHistory);
    connect(&m_client, &TradingClient::candleReceived, this, &Application::handleCandle);

    connect(&m_client, &TradingClient::connectionStateChanged, this,
            [this](const QString& status) {
                m_connectionStatus = status;
                Q_EMIT connectionStatusChanged();
            });

    connect(&m_client, &TradingClient::performanceGuardUpdated, this,
            [this](const PerformanceGuard& guard) {
                m_guard = guard;
                Q_EMIT performanceGuardChanged();
                if (m_frameMonitor) {
                    m_frameMonitor->setPerformanceGuard(m_guard);
                }
            });

    connect(&m_client, &TradingClient::streamingChanged, this, [this]() {
        const QString state = m_client.isStreaming()
                                  ? QStringLiteral("streaming")
                                  : QStringLiteral("idle");
        m_connectionStatus = state;
        Q_EMIT connectionStatusChanged();
    });
}

QString Application::instrumentLabel() const {
    return m_instrument.symbol;
}

void Application::configureParser(QCommandLineParser& parser) const {
    parser.addHelpOption();
    parser.addOption({{"e", "endpoint"}, tr("Adres gRPC host:port"), tr("endpoint"),
                      QStringLiteral("127.0.0.1:50061")});
    parser.addOption({"exchange", tr("Nazwa giełdy"), tr("exchange"), QStringLiteral("BINANCE")});
    parser.addOption({"symbol", tr("Symbol logiczny"), tr("symbol"), QStringLiteral("BTC/USDT")});
    parser.addOption({"venue-symbol", tr("Symbol na giełdzie"), tr("venue"),
                      QStringLiteral("BTCUSDT")});
    parser.addOption({"quote", tr("Waluta kwotowana"), tr("quote"), QStringLiteral("USDT")});
    parser.addOption({"base", tr("Waluta bazowa"), tr("base"), QStringLiteral("BTC")});
    parser.addOption({"granularity", tr("ISO-8601 duration"), tr("granularity"),
                      QStringLiteral("PT1M")});
    parser.addOption({"history-limit", tr("Limit pobieranej historii"), tr("limit"),
                      QStringLiteral("500")});
    parser.addOption({"max-samples", tr("Maksymalna liczba świec w modelu"), tr("samples"),
                      QStringLiteral("10240")});
    parser.addOption({"fps-target", tr("Docelowy FPS"), tr("fps"), QStringLiteral("60")});
    parser.addOption({"reduce-motion-after", tr("Czas (s) po którym ograniczamy animacje"),
                      tr("seconds"), QStringLiteral("1.0")});
    parser.addOption({"jank-threshold-ms", tr("Budżet janku w ms"), tr("ms"),
                      QStringLiteral("18.0")});
    parser.addOption({"max-overlay-count", tr("Limit nakładek na wykres"), tr("count"),
                      QStringLiteral("3")});
    parser.addOption({"overlay-disable-secondary-fps",
                      tr("Próg FPS wyłączający nakładki drugorzędne"), tr("fps"),
                      QStringLiteral("0")});
    parser.addOption({"metrics-endpoint", tr("Adres serwera MetricsService"), tr("endpoint"),
                      QStringLiteral("127.0.0.1:50061")});
    parser.addOption({"metrics-tag", tr("Etykieta notatek telemetrii"), tr("tag"), QString()});
    parser.addOption({"disable-metrics", tr("Wyłącza wysyłkę telemetrii")});
    parser.addOption({"metrics-use-tls", tr("Wymusza połączenie TLS z MetricsService")});
    parser.addOption({"metrics-root-cert", tr("Plik root CA (PEM)"), tr("path"), QString()});
    parser.addOption({"metrics-client-cert", tr("Certyfikat klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"metrics-client-key", tr("Klucz klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"metrics-server-name", tr("Override nazwy serwera TLS"), tr("name"), QString()});
    parser.addOption({"metrics-server-sha256", tr("Oczekiwany odcisk SHA-256 certyfikatu serwera"), tr("hex"),
                      QString()});
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
    m_instrument = instrument;
    Q_EMIT instrumentChanged();

    const int historyLimit = parser.value("history-limit").toInt();
    m_client.setHistoryLimit(historyLimit);

    m_maxSamples = parser.value("max-samples").toInt();
    if (m_maxSamples <= 0)
        m_maxSamples = 10240;

    PerformanceGuard guard;
    guard.fpsTarget = parser.value("fps-target").toInt();
    guard.reduceMotionAfterSeconds = parser.value("reduce-motion-after").toDouble();
    guard.jankThresholdMs = parser.value("jank-threshold-ms").toDouble();
    guard.maxOverlayCount = parser.value("max-overlay-count").toInt();
    guard.disableSecondaryWhenFpsBelow = parser.value("overlay-disable-secondary-fps").toInt();

    m_client.setPerformanceGuard(guard);
    m_guard = guard;
    Q_EMIT performanceGuardChanged();

    if (m_frameMonitor) {
        m_frameMonitor->setPerformanceGuard(m_guard);
    }

    m_metricsEndpoint = parser.value("metrics-endpoint");
    m_metricsTag = parser.value("metrics-tag");
    m_metricsEnabled = !parser.isSet("disable-metrics");

    TelemetryTlsConfig tlsConfig;
    tlsConfig.enabled = parser.isSet("metrics-use-tls");
    tlsConfig.rootCertificatePath = parser.value("metrics-root-cert");
    tlsConfig.clientCertificatePath = parser.value("metrics-client-cert");
    tlsConfig.clientKeyPath = parser.value("metrics-client-key");
    tlsConfig.serverNameOverride = parser.value("metrics-server-name");
    tlsConfig.pinnedServerSha256 = parser.value("metrics-server-sha256");
    m_tlsConfig = tlsConfig;
    ensureTelemetry();

    return true;
}

void Application::start() {
    m_ohlcvModel.setMaximumSamples(m_maxSamples);
    ensureFrameMonitor();

    // Jeśli QML już wczytany — podepnij okno
    if (!m_engine.rootObjects().isEmpty()) {
        attachWindow(m_engine.rootObjects().constFirst());
    }

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

void Application::ensureFrameMonitor() {
    if (m_frameMonitor)
        return;

    m_frameMonitor = std::make_unique<FrameRateMonitor>(this);
    connect(m_frameMonitor.get(), &FrameRateMonitor::reduceMotionSuggested, this,
            [this](bool enabled) {
                if (m_reduceMotionActive == enabled) {
                    reportReduceMotionTelemetry(enabled);
                    return;
                }
                m_reduceMotionActive = enabled;
                Q_EMIT reduceMotionActiveChanged();
                reportReduceMotionTelemetry(enabled);
            });
    connect(m_frameMonitor.get(), &FrameRateMonitor::frameSampled, this, [this](double fps) {
        m_latestFpsSample = fps;
        if (m_pendingReduceMotionState.has_value()) {
            const bool pending = m_pendingReduceMotionState.value();
            m_pendingReduceMotionState.reset();
            reportReduceMotionTelemetry(pending);
        }
    });
    m_frameMonitor->setPerformanceGuard(m_guard);
}

void Application::attachWindow(QObject* object) {
    auto* window = qobject_cast<QQuickWindow*>(object);
    if (!window && object) {
        window = object->findChild<QQuickWindow*>();
    }
    if (!window)
        return;

    ensureFrameMonitor();
    m_frameMonitor->setWindow(window);
}

void Application::setTelemetryReporter(std::unique_ptr<TelemetryReporter> reporter) {
    m_telemetry = std::move(reporter);
    ensureTelemetry();
}

void Application::notifyOverlayUsage(int activeCount, int allowedCount, bool reduceMotionActive) {
    OverlayState state;
    state.active = qMax(0, activeCount);
    state.allowed = qMax(0, allowedCount);
    state.reduceMotion = reduceMotionActive;
    m_lastOverlayState = state;
    reportOverlayTelemetry();
}

void Application::notifyWindowCount(int totalWindowCount) {
    m_windowCount = qMax(1, totalWindowCount);
    ensureTelemetry();
    if (m_telemetry) {
        m_telemetry->setWindowCount(m_windowCount);
    }
}

void Application::ingestFpsSampleForTesting(double fps) {
    m_latestFpsSample = fps;
    if (m_pendingReduceMotionState.has_value()) {
        const bool pending = m_pendingReduceMotionState.value();
        m_pendingReduceMotionState.reset();
        reportReduceMotionTelemetry(pending);
    }
}

void Application::setReduceMotionStateForTesting(bool active) {
    m_reduceMotionActive = active;
    reportReduceMotionTelemetry(active);
}

void Application::ensureTelemetry() {
    const bool shouldEnable = m_metricsEnabled && !m_metricsEndpoint.isEmpty();
    if (!m_telemetry) {
        if (!shouldEnable)
            return;
        auto reporter = std::make_unique<UiTelemetryReporter>(this);
        m_telemetry = std::move(reporter);
    }
    if (!m_telemetry)
        return;

    m_telemetry->setWindowCount(m_windowCount);
    if (!m_metricsTag.isEmpty())
        m_telemetry->setNotesTag(m_metricsTag);
    if (!m_metricsEndpoint.isEmpty())
        m_telemetry->setEndpoint(m_metricsEndpoint);
    m_telemetry->setTlsConfig(m_tlsConfig);
    m_telemetry->setEnabled(shouldEnable);
}

void Application::reportOverlayTelemetry() {
    ensureTelemetry();
    if (!m_telemetry || !m_telemetry->isEnabled() || !m_lastOverlayState)
        return;

    const OverlayState current = *m_lastOverlayState;
    if (m_lastOverlayTelemetryReported && *m_lastOverlayTelemetryReported == current)
        return;

    m_telemetry->reportOverlayBudget(m_guard, current.active, current.allowed, current.reduceMotion);
    m_lastOverlayTelemetryReported = current;
}

void Application::reportReduceMotionTelemetry(bool enabled) {
    ensureTelemetry();
    if (!m_telemetry || !m_telemetry->isEnabled()) {
        m_pendingReduceMotionState.reset();
        return;
    }

    if (m_latestFpsSample <= 0.0) {
        m_pendingReduceMotionState = enabled;
        return;
    }

    if (m_lastReduceMotionReported && m_lastReduceMotionReported.value() == enabled)
        return;

    OverlayState overlay = m_lastOverlayState.value_or(OverlayState{});
    if (overlay.allowed == 0) {
        overlay.allowed = m_guard.maxOverlayCount > 0 ? m_guard.maxOverlayCount : 0;
    }

    m_telemetry->reportReduceMotion(m_guard, enabled, m_latestFpsSample, overlay.active, overlay.allowed);
    m_lastReduceMotionReported = enabled;
    m_pendingReduceMotionState.reset();
}
