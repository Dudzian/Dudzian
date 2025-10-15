#include "Application.hpp"

#include <QQmlContext>
#include <QQuickWindow>
#include <QtGlobal>
#include <QDir>
#include <QFile>
#include <QIODevice>
#include <QLoggingCategory>
#include <QDebug>
#include <QByteArray>
#include <QGuiApplication>
#include <QPoint>
#include <QRect>
#include <QScreen>
#include <QCommandLineParser>
#include <optional>

#include "telemetry/TelemetryReporter.hpp"
#include "telemetry/UiTelemetryReporter.hpp"
#include "utils/FrameRateMonitor.hpp"
#include "license/LicenseActivationController.hpp"
#include "app/ActivationController.hpp"

Q_LOGGING_CATEGORY(lcAppMetrics, "bot.shell.app.metrics")

namespace {

std::optional<QString> envValue(const QByteArray& key)
{
    if (!qEnvironmentVariableIsSet(key.constData()))
        return std::nullopt;
    return qEnvironmentVariable(key.constData());
}

std::optional<bool> envBool(const QByteArray& key)
{
    const auto valueOpt = envValue(key);
    if (!valueOpt.has_value())
        return std::nullopt;
    const QString normalized = valueOpt->trimmed().toLower();
    if (normalized.isEmpty())
        return std::nullopt;
    if (normalized == QStringLiteral("1") || normalized == QStringLiteral("true") ||
        normalized == QStringLiteral("yes") || normalized == QStringLiteral("on"))
        return true;
    if (normalized == QStringLiteral("0") || normalized == QStringLiteral("false") ||
        normalized == QStringLiteral("no") || normalized == QStringLiteral("off"))
        return false;
    qCWarning(lcAppMetrics) << "Nieprawidłowa wartość" << *valueOpt
                            << "w zmiennej" << QString::fromUtf8(key)
                            << "– oczekiwano wartości boolowskiej (true/false).";
    return std::nullopt;
}

QString expandUserPath(QString path)
{
    if (!path.startsWith(QLatin1Char('~')))
        return path;
    if (path == QStringLiteral("~"))
        return QDir::homePath();
    if (path.startsWith(QStringLiteral("~/")))
        return QDir::homePath() + path.mid(1);
    return path;
}

QString readTokenFile(const QString& rawPath)
{
    if (rawPath.trimmed().isEmpty())
        return {};
    const QString path = expandUserPath(rawPath.trimmed());
    QFile file(path);
    if (!file.exists()) {
        qCWarning(lcAppMetrics) << "Plik z tokenem MetricsService nie istnieje:" << path;
        return {};
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCWarning(lcAppMetrics) << "Nie udało się odczytać pliku z tokenem" << path
                                << file.errorString();
        return {};
    }
    const QByteArray data = file.readAll();
    QString token = QString::fromUtf8(data).trimmed();
    if (token.isEmpty()) {
        qCWarning(lcAppMetrics) << "Plik" << path << "nie zawiera tokenu autoryzacyjnego MetricsService";
        return {};
    }
    return token;
}

} // namespace

Application::Application(QQmlApplicationEngine& engine, QObject* parent)
    : QObject(parent)
    , m_engine(engine)
{
    m_activationController = std::make_unique<ActivationController>(this);

    // Startowe ustawienia instrumentu z klienta (mogą być nadpisane przez CLI)
    m_instrument = m_client.instrumentConfig();

    m_licenseController = std::make_unique<LicenseActivationController>();
    m_licenseController->setParent(this);
    m_licenseController->setConfigDirectory(QDir::current().absoluteFilePath(QStringLiteral("config")));
    m_licenseController->setLicenseStoragePath(QDir::current().absoluteFilePath(QStringLiteral("var/licenses/active/license.json")));

    exposeToQml();

    // Podłącz okno po utworzeniu (dla FrameRateMonitor)
    connect(&m_engine, &QQmlApplicationEngine::objectCreated, this,
            [this](QObject* object, const QUrl&) { attachWindow(object); });

    // Połączenia sygnałów klienta (market data)
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
                // Telemetria overlay może zależeć od nowych limitów
                reportOverlayTelemetry();
            });

    connect(&m_client, &TradingClient::streamingChanged, this, [this]() {
        const QString state = m_client.isStreaming()
                                  ? QStringLiteral("streaming")
                                  : QStringLiteral("idle");
        m_connectionStatus = state;
        Q_EMIT connectionStatusChanged();
    });

    // Risk state (jeśli dostępne po stronie serwera)
    connect(&m_client, &TradingClient::riskStateReceived, this, &Application::handleRiskState);
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

    parser.addOption({"screen-name", tr("Preferowany ekran (nazwa QScreen)"), tr("name")});
    parser.addOption({"screen-index", tr("Preferowany ekran (indeks)"), tr("index")});
    parser.addOption({"primary-screen", tr("Wymusza użycie ekranu podstawowego")});

    // Telemetria MetricsService
    parser.addOption({"metrics-endpoint", tr("Adres serwera MetricsService"), tr("endpoint"),
                      QStringLiteral("127.0.0.1:50061")});
    parser.addOption({"metrics-tag", tr("Etykieta notatek telemetrii"), tr("tag"), QString()});
    parser.addOption({"metrics-auth-token", tr("Token autoryzacyjny MetricsService"), tr("token")});
    parser.addOption({"metrics-auth-token-file", tr("Ścieżka pliku z tokenem MetricsService"), tr("path")});
    parser.addOption({"disable-metrics", tr("Wyłącza wysyłkę telemetrii")});
    parser.addOption({"no-metrics", tr("Alias: wyłącza wysyłkę telemetrii")});

    // TLS/mTLS gRPC (demon tradingowy)
    parser.addOption({"grpc-use-mtls", tr("Wymusza mTLS dla klienta tradingowego")});
    parser.addOption({"grpc-root-cert", tr("Root CA (PEM) dla kanału tradingowego"), tr("path"), QString()});
    parser.addOption({"grpc-client-cert", tr("Certyfikat klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"grpc-client-key", tr("Klucz klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"grpc-target-name", tr("Override nazwy hosta TLS"), tr("name"), QString()});

    // TLS/mTLS dla TradingClient (ogólne przełączniki – mogą być nadpisane przez --grpc-*)
    parser.addOption({"use-tls", tr("Wymusza połączenie TLS z TradingService")});
    parser.addOption({"tls-root-cert", tr("Plik root CA (PEM) dla TradingService"), tr("path"), QString()});
    parser.addOption({"tls-client-cert", tr("Certyfikat klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"tls-client-key", tr("Klucz klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"tls-server-name", tr("Override nazwy serwera TLS"), tr("name"), QString()});
    parser.addOption({"tls-pinned-sha256", tr("Oczekiwany fingerprint SHA-256 certyfikatu"), tr("hex"), QString()});
    parser.addOption({"tls-require-client-auth", tr("Wymaga dostarczenia certyfikatu klienta (mTLS)")});

    // TLS/mTLS dla MetricsService (opcjonalnie)
    parser.addOption({"metrics-use-tls", tr("Wymusza połączenie TLS z MetricsService")});
    parser.addOption({"metrics-root-cert", tr("Plik root CA (PEM)"), tr("path"), QString()});
    parser.addOption({"metrics-client-cert", tr("Certyfikat klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"metrics-client-key", tr("Klucz klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"metrics-server-name", tr("Override nazwy serwera TLS"), tr("name"), QString()});
    parser.addOption({"metrics-server-sha256", tr("Oczekiwany odcisk SHA-256 certyfikatu serwera"), tr("hex"),
                      QString()});

    // Licencje OEM
    parser.addOption({"license-storage", tr("Ścieżka zapisu aktywowanej licencji OEM"), tr("path"),
                      QStringLiteral("var/licenses/active/license.json")});
    parser.addOption({"expected-fingerprint-path", tr("Ścieżka oczekiwanego fingerprintu OEM (JSON/tekst)"), tr("path"),
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

    // Ogólny TLS (może być nadpisany przez sekcję gRPC)
    TradingClient::TlsConfig tlsConfig;
    tlsConfig.enabled = parser.isSet("use-tls");
    tlsConfig.rootCertificatePath = parser.value("tls-root-cert");
    tlsConfig.clientCertificatePath = parser.value("tls-client-cert");
    tlsConfig.clientKeyPath = parser.value("tls-client-key");
    tlsConfig.serverNameOverride = parser.value("tls-server-name");
    tlsConfig.pinnedServerFingerprint = parser.value("tls-pinned-sha256");
    tlsConfig.requireClientAuth = parser.isSet("tls-require-client-auth");
    m_client.setTlsConfig(tlsConfig);

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

    m_forcePrimaryScreen = false;
    m_preferredScreenName.clear();
    m_preferredScreenIndex = -1;
    m_screenWarningLogged = false;

    m_forcePrimaryScreen = parser.isSet("primary-screen");
    m_preferredScreenName = parser.value("screen-name").trimmed();
    if (!parser.value("screen-index").trimmed().isEmpty()) {
        bool ok = false;
        const int index = parser.value("screen-index").toInt(&ok);
        if (ok && index >= 0) {
            m_preferredScreenIndex = index;
        } else if (parser.isSet("screen-index")) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowy indeks ekranu" << parser.value("screen-index")
                << "– oczekiwano liczby całkowitej >= 0.";
            m_preferredScreenIndex = -1;
        }
    } else if (parser.isSet("screen-index")) {
        m_preferredScreenIndex = -1;
    }

    applyScreenEnvironmentOverrides(parser);
    m_preferredScreenConfigured = m_forcePrimaryScreen || m_preferredScreenIndex >= 0
        || !m_preferredScreenName.isEmpty();

    // gRPC TLS – nadpisuje ogólny TLS jeśli podany
    TradingClient::TlsConfig tradingTls;
    tradingTls.enabled = parser.isSet("grpc-use-mtls");
    const QString cliRootCert = parser.value("grpc-root-cert").trimmed();
    if (!cliRootCert.isEmpty())
        tradingTls.rootCertificatePath = expandUserPath(cliRootCert);
    const QString cliClientCert = parser.value("grpc-client-cert").trimmed();
    if (!cliClientCert.isEmpty())
        tradingTls.clientCertificatePath = expandUserPath(cliClientCert);
    const QString cliClientKey = parser.value("grpc-client-key").trimmed();
    if (!cliClientKey.isEmpty())
        tradingTls.clientKeyPath = expandUserPath(cliClientKey);
    tradingTls.targetNameOverride = parser.value("grpc-target-name");
    m_tradingTlsConfig = tradingTls;

    // --- Telemetria ---
    m_metricsEndpoint = parser.value("metrics-endpoint");
    if (m_metricsEndpoint.isEmpty()) {
        // Fallback: użyj endpointu tradingowego, jeśli nie podano dedykowanego dla MetricsService
        m_metricsEndpoint = parser.value("endpoint");
    }
    m_metricsTag = parser.value("metrics-tag");
    m_metricsEnabled = !(parser.isSet("disable-metrics") || parser.isSet("no-metrics"));

    QString cliToken = parser.value("metrics-auth-token").trimmed();
    QString cliTokenFile = parser.value("metrics-auth-token-file").trimmed();
    const bool cliTokenProvided = !cliToken.isEmpty();
    const bool cliTokenFileProvided = !cliTokenFile.isEmpty();
    if (cliTokenProvided && cliTokenFileProvided) {
        qCWarning(lcAppMetrics)
            << "Podano jednocześnie --metrics-auth-token oraz --metrics-auth-token-file. Użyję tokenu przekazanego bezpośrednio.";
    }

    if (cliTokenProvided) {
        m_metricsAuthToken = cliToken;
    } else if (cliTokenFileProvided) {
        m_metricsAuthToken = readTokenFile(cliTokenFile);
    } else {
        m_metricsAuthToken.clear();
    }

    applyTradingTlsEnvironmentOverrides(parser);
    m_client.setTlsConfig(m_tradingTlsConfig);

    // TLS config (MetricsService)
    TelemetryTlsConfig mtls;
    mtls.enabled = parser.isSet("metrics-use-tls");
    mtls.rootCertificatePath = parser.value("metrics-root-cert");
    mtls.clientCertificatePath = parser.value("metrics-client-cert");
    mtls.clientKeyPath = parser.value("metrics-client-key");
    mtls.serverNameOverride = parser.value("metrics-server-name");
    mtls.pinnedServerSha256 = parser.value("metrics-server-sha256");
    m_tlsConfig = mtls;

    // Konfiguracja licencji OEM (CLI/ENV)
    const QString cliLicensePath = parser.value("license-storage").trimmed();
    if (!cliLicensePath.isEmpty())
        m_licenseController->setLicenseStoragePath(expandUserPath(cliLicensePath));
    const QString cliExpectedFingerprint = parser.value("expected-fingerprint-path").trimmed();
    if (!cliExpectedFingerprint.isEmpty())
        m_licenseController->setFingerprintDocumentPath(expandUserPath(cliExpectedFingerprint));
    m_licenseController->initialize();

    applyMetricsEnvironmentOverrides(parser, cliTokenProvided, cliTokenFileProvided);

    // Inicjalizacja/reportera + token
    ensureTelemetry();

    return true;
}

void Application::start() {
    m_ohlcvModel.setMaximumSamples(m_maxSamples);
    m_riskModel.clear();

    ensureFrameMonitor();

    // Jeśli QML już wczytany — podepnij okno
    if (!m_engine.rootObjects().isEmpty()) {
        attachWindow(m_engine.rootObjects().constFirst());
    }

    ensureTelemetry();
    if (m_telemetry) {
        m_telemetry->setWindowCount(m_windowCount);
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

void Application::handleRiskState(const RiskSnapshotData& snapshot) {
    m_riskModel.updateFromSnapshot(snapshot);
}

void Application::exposeToQml() {
    m_engine.rootContext()->setContextProperty(QStringLiteral("appController"), this);
    m_engine.rootContext()->setContextProperty(QStringLiteral("ohlcvModel"), &m_ohlcvModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("riskModel"), &m_riskModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("licenseController"), m_licenseController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("activationController"), m_activationController.get());
}

QObject* Application::activationController() const
{
    return m_activationController.get();
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
                m_pendingReduceMotionState = enabled;  // zapamiętaj do telemetrii
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

    connect(m_frameMonitor.get(), &FrameRateMonitor::jankBudgetBreached, this,
            [this](double frameMs, double thresholdMs) { reportJankTelemetry(frameMs, thresholdMs); });

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
    applyPreferredScreen(window);
    updateScreenInfo(window->screen());
    connect(window, &QQuickWindow::screenChanged, this,
            [this](QScreen* screen) { updateScreenInfo(screen); },
            Qt::UniqueConnection);
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

void Application::applyTradingTlsEnvironmentOverrides(const QCommandLineParser& parser)
{
    const bool cliTlsEnabled = parser.isSet("grpc-use-mtls");
    if (!cliTlsEnabled) {
        if (const auto tlsEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_GRPC_USE_MTLS")); tlsEnv.has_value())
            m_tradingTlsConfig.enabled = tlsEnv.value();
    }

    auto applyPathFromEnv = [&](const QByteArray& key, const QString& cliValue, QString TradingClient::TlsConfig::*field) {
        if (!cliValue.trimmed().isEmpty())
            return;
        if (const auto value = envValue(key)) {
            const QString trimmed = value->trimmed();
            if (trimmed.compare(QStringLiteral("NONE"), Qt::CaseInsensitive) == 0
                || trimmed.compare(QStringLiteral("NULL"), Qt::CaseInsensitive) == 0) {
                m_tradingTlsConfig.*field = QString();
            } else {
                m_tradingTlsConfig.*field = expandUserPath(trimmed);
            }
        }
    };

    applyPathFromEnv(QByteArrayLiteral("BOT_CORE_UI_GRPC_ROOT_CERT"), parser.value("grpc-root-cert"),
                     &TradingClient::TlsConfig::rootCertificatePath);
    applyPathFromEnv(QByteArrayLiteral("BOT_CORE_UI_GRPC_CLIENT_CERT"), parser.value("grpc-client-cert"),
                     &TradingClient::TlsConfig::clientCertificatePath);
    applyPathFromEnv(QByteArrayLiteral("BOT_CORE_UI_GRPC_CLIENT_KEY"), parser.value("grpc-client-key"),
                     &TradingClient::TlsConfig::clientKeyPath);

    if (parser.value("grpc-target-name").trimmed().isEmpty()) {
        if (const auto targetEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_TARGET_NAME")))
            m_tradingTlsConfig.targetNameOverride = targetEnv->trimmed();
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

void Application::simulateFrameIntervalForTesting(double seconds) {
    ensureFrameMonitor();
    if (!m_frameMonitor)
        return;
    m_frameMonitor->simulateFrameIntervalForTest(seconds);
}

void Application::applyMetricsEnvironmentOverrides(const QCommandLineParser& parser,
                                                    bool cliTokenProvided,
                                                    bool cliTokenFileProvided) {
    if (const auto endpointEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_ENDPOINT"));
        endpointEnv.has_value()) {
        m_metricsEndpoint = endpointEnv->trimmed();
    }

    if (const auto tagEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_TAG")); tagEnv.has_value()) {
        m_metricsTag = tagEnv->trimmed();
    }

    const bool disableFlagsProvided = parser.isSet("disable-metrics") || parser.isSet("no-metrics");
    if (!disableFlagsProvided) {
        if (const auto enabledEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_METRICS_ENABLED")); enabledEnv.has_value()) {
            m_metricsEnabled = enabledEnv.value();
        }
        if (const auto disabledEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_METRICS_DISABLED")); disabledEnv.has_value()) {
            m_metricsEnabled = !disabledEnv.value();
        }
    }

    bool envTlsExplicit = false;
    if (!parser.isSet("metrics-use-tls")) {
        if (const auto tlsEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_METRICS_USE_TLS")); tlsEnv.has_value()) {
            m_tlsConfig.enabled = tlsEnv.value();
            envTlsExplicit = true;
        }
    }

    bool tlsMaterialProvided = false;
    auto overrideString = [&](QString TelemetryTlsConfig::*field, const QByteArray& key) {
        if (const auto value = envValue(key); value.has_value()) {
            const QString trimmed = value->trimmed();
            if (trimmed.isEmpty()) {
                m_tlsConfig.*field = QString();
            } else {
                m_tlsConfig.*field = expandUserPath(trimmed);
                tlsMaterialProvided = true;
            }
        }
    };

    overrideString(&TelemetryTlsConfig::rootCertificatePath, QByteArrayLiteral("BOT_CORE_UI_METRICS_ROOT_CERT"));
    overrideString(&TelemetryTlsConfig::clientCertificatePath, QByteArrayLiteral("BOT_CORE_UI_METRICS_CLIENT_CERT"));
    overrideString(&TelemetryTlsConfig::clientKeyPath, QByteArrayLiteral("BOT_CORE_UI_METRICS_CLIENT_KEY"));
    overrideString(&TelemetryTlsConfig::serverNameOverride, QByteArrayLiteral("BOT_CORE_UI_METRICS_SERVER_NAME"));

    if (const auto shaEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_SERVER_SHA256")); shaEnv.has_value()) {
        m_tlsConfig.pinnedServerSha256 = shaEnv->trimmed();
        if (!m_tlsConfig.pinnedServerSha256.isEmpty())
            tlsMaterialProvided = true;
    }

    if (tlsMaterialProvided && !m_tlsConfig.enabled && !envTlsExplicit && !parser.isSet("metrics-use-tls")) {
        qCDebug(lcAppMetrics) << "Wymuszam TLS dla telemetrii na podstawie zmiennych środowiskowych.";
        m_tlsConfig.enabled = true;
    }

    if (!cliTokenProvided && !cliTokenFileProvided) {
        const auto envToken = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_AUTH_TOKEN"));
        const auto envTokenFile = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_AUTH_TOKEN_FILE"));

        const bool envTokenNonEmpty = envToken.has_value() && !envToken->trimmed().isEmpty();
        const bool envTokenFileNonEmpty = envTokenFile.has_value() && !envTokenFile->trimmed().isEmpty();

        if (envTokenNonEmpty && envTokenFileNonEmpty) {
            qCWarning(lcAppMetrics)
                << "Zmiennie BOT_CORE_UI_METRICS_AUTH_TOKEN oraz BOT_CORE_UI_METRICS_AUTH_TOKEN_FILE są ustawione równocześnie."
                << "Użyję wartości tokenu przekazanej w zmiennej BOT_CORE_UI_METRICS_AUTH_TOKEN.";
        }

        bool applied = false;
        if (envToken.has_value()) {
            const QString trimmed = envToken->trimmed();
            if (!trimmed.isEmpty()) {
                m_metricsAuthToken = trimmed;
                applied = true;
            }
        }

        if (!applied && envTokenFileNonEmpty) {
            const QString tokenFromFile = readTokenFile(envTokenFile->trimmed());
            if (!tokenFromFile.isEmpty()) {
                m_metricsAuthToken = tokenFromFile;
                applied = true;
            }
        }

        if (!applied && ((envToken.has_value() && envToken->trimmed().isEmpty()) ||
                         (envTokenFile.has_value() && envTokenFile->trimmed().isEmpty()))) {
            m_metricsAuthToken.clear();
        }
    }
}

void Application::applyScreenEnvironmentOverrides(const QCommandLineParser& parser)
{
    if (!parser.isSet("screen-name")) {
        if (const auto nameEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_SCREEN_NAME"));
            nameEnv.has_value()) {
            const QString trimmed = nameEnv->trimmed();
            if (trimmed.isEmpty()) {
                m_preferredScreenName.clear();
            } else {
                m_preferredScreenName = trimmed;
            }
        }
    }

    if (!parser.isSet("screen-index")) {
        if (const auto indexEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_SCREEN_INDEX"));
            indexEnv.has_value()) {
            const QString trimmed = indexEnv->trimmed();
            if (trimmed.isEmpty()) {
                m_preferredScreenIndex = -1;
            } else {
                bool ok = false;
                const int value = trimmed.toInt(&ok);
                if (ok && value >= 0) {
                    m_preferredScreenIndex = value;
                } else {
                    qCWarning(lcAppMetrics)
                        << "Nieprawidłowa wartość" << trimmed
                        << "w zmiennej BOT_CORE_UI_SCREEN_INDEX – oczekiwano liczby całkowitej >= 0.";
                }
            }
        }
    }

    if (!parser.isSet("primary-screen")) {
        if (const auto primaryEnv = envBool(QByteArrayLiteral("BOT_CORE_UI_SCREEN_PRIMARY"));
            primaryEnv.has_value()) {
            m_forcePrimaryScreen = primaryEnv.value();
        }
    }
}

void Application::applyPreferredScreen(QQuickWindow* window)
{
    if (!window)
    {
        updateScreenInfo(nullptr);
        return;
    }

    QScreen* target = resolvePreferredScreen();
    if (target && window->screen() != target) {
        window->setScreen(target);
        const QRect geometry = target->geometry();
        const QPoint desired = geometry.center() - QPoint(window->width() / 2, window->height() / 2);
        window->setPosition(desired);
    }

    QScreen* effectiveScreen = window->screen();
    if (!effectiveScreen)
        effectiveScreen = target;
    updateScreenInfo(effectiveScreen);
}

QScreen* Application::resolvePreferredScreen() const
{
    const QList<QScreen*> screens = QGuiApplication::screens();
    if (screens.isEmpty())
        return nullptr;

    if (m_forcePrimaryScreen) {
        if (auto* primary = QGuiApplication::primaryScreen())
            return primary;
    }

    if (m_preferredScreenIndex >= 0) {
        if (m_preferredScreenIndex < screens.size()) {
            return screens.at(m_preferredScreenIndex);
        }
        if (m_preferredScreenConfigured && !m_screenWarningLogged) {
            qCWarning(lcAppMetrics)
                << "Żądany indeks ekranu" << m_preferredScreenIndex
                << "wykracza poza dostępne ekrany (liczba ekranów:" << screens.size() << ").";
            m_screenWarningLogged = true;
        }
    }

    if (!m_preferredScreenName.trimmed().isEmpty()) {
        const QString normalized = m_preferredScreenName.trimmed().toLower();
        for (QScreen* screen : screens) {
            if (!screen)
                continue;
            const QString screenName = screen->name();
            if (screenName.compare(normalized, Qt::CaseInsensitive) == 0
                || screenName.toLower().contains(normalized)) {
                return screen;
            }
        }
        if (m_preferredScreenConfigured && !m_screenWarningLogged) {
            qCWarning(lcAppMetrics)
                << "Nie znaleziono ekranu o nazwie zawierającej" << m_preferredScreenName << '.';
            m_screenWarningLogged = true;
        }
    }

    if (m_forcePrimaryScreen) {
        if (auto* primary = QGuiApplication::primaryScreen())
            return primary;
    }

    return nullptr;
}

void Application::applyPreferredScreenForTesting(QQuickWindow* window)
{
    applyPreferredScreen(window);
}

QScreen* Application::pickPreferredScreenForTesting() const
{
    return resolvePreferredScreen();
}

void Application::updateScreenInfo(QScreen* screen)
{
    if (!screen) {
        m_screenInfo.reset();
        if (m_telemetry) {
            m_telemetry->clearScreenInfo();
        }
        return;
    }

    TelemetryReporter::ScreenInfo info;
    info.name = screen->name();
    info.manufacturer = screen->manufacturer();
    info.model = screen->model();
    info.serialNumber = screen->serialNumber();
    info.geometry = screen->geometry();
    info.availableGeometry = screen->availableGeometry();
    info.refreshRateHz = screen->refreshRate();
    info.devicePixelRatio = screen->devicePixelRatio();
    info.logicalDpiX = screen->logicalDotsPerInchX();
    info.logicalDpiY = screen->logicalDotsPerInchY();
    const QList<QScreen*> screens = QGuiApplication::screens();
    info.index = screens.indexOf(screen);

    m_screenInfo = info;
    if (m_telemetry) {
        m_telemetry->setScreenInfo(info);
    }
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
    m_telemetry->setNotesTag(m_metricsTag);
    m_telemetry->setEndpoint(m_metricsEndpoint);
    m_telemetry->setTlsConfig(m_tlsConfig);
    m_telemetry->setAuthToken(m_metricsAuthToken);
    if (m_screenInfo.has_value()) {
        m_telemetry->setScreenInfo(m_screenInfo.value());
    } else {
        m_telemetry->clearScreenInfo();
    }
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

void Application::reportJankTelemetry(double frameMs, double thresholdMs) {
    if (frameMs <= 0.0)
        return;

    ensureTelemetry();
    if (!m_telemetry || !m_telemetry->isEnabled())
        return;

    if (!m_jankTelemetryTimerValid) {
        m_lastJankTelemetry.start();
        m_jankTelemetryTimerValid = true;
    } else if (m_lastJankTelemetry.isValid() && m_lastJankTelemetry.elapsed() < m_jankTelemetryCooldownMs) {
        return;
    } else if (m_lastJankTelemetry.isValid()) {
        m_lastJankTelemetry.restart();
    } else {
        m_lastJankTelemetry.start();
    }

    OverlayState overlay = m_lastOverlayState.value_or(OverlayState{});
    if (overlay.allowed == 0) {
        overlay.allowed = m_guard.maxOverlayCount > 0 ? m_guard.maxOverlayCount : 0;
    }

    m_telemetry->reportJankEvent(
        m_guard,
        frameMs,
        thresholdMs,
        m_reduceMotionActive,
        overlay.active,
        overlay.allowed
    );
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
