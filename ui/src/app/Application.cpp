#include "Application.hpp"

#include <QByteArray>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QGuiApplication>
#include <QIODevice>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QStringList>
#include <QPoint>
#include <QRect>
#include <QQmlContext>
#include <QQuickWindow>
#include <QSaveFile>
#include <QScreen>
#include <QHash>
#include <QPair>
#include <QRegularExpression>
#include <QSysInfo>
#include <QSet>
#include <QUrl>
#include <QTimer>
#include <QScopeGuard>
#include <QtGlobal>
#include <optional>
#include <cmath>
#include <algorithm>

#include "telemetry/TelemetryReporter.hpp"
#include "telemetry/UiTelemetryReporter.hpp"
#include "utils/FrameRateMonitor.hpp"
#include "license/LicenseActivationController.hpp"
#include "app/ActivationController.hpp"
#include "app/StrategyConfigController.hpp"
#include "security/SecurityAdminController.hpp"
#include "support/SupportBundleController.hpp"
#include "reporting/ReportCenterController.hpp"
#include "grpc/BotCoreLocalService.hpp"

Q_LOGGING_CATEGORY(lcAppMetrics, "bot.shell.app.metrics")

namespace {

constexpr double kDefaultRiskRefreshSeconds = 5.0;
constexpr int kMinRiskRefreshIntervalMs = 1000;
constexpr int kMaxRiskRefreshIntervalMs = 300000;
constexpr int kUiSettingsDebounceMs = 500;
constexpr auto kUiSettingsEnv = QByteArrayLiteral("BOT_CORE_UI_SETTINGS_PATH");
constexpr auto kUiSettingsDisableEnv = QByteArrayLiteral("BOT_CORE_UI_SETTINGS_DISABLE");
constexpr auto kRiskHistoryExportDirEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_EXPORT_DIR");
constexpr auto kRiskHistoryExportLimitEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_EXPORT_LIMIT");
constexpr auto kRiskHistoryExportLimitEnabledEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_EXPORT_LIMIT_ENABLED");
constexpr auto kRiskHistoryAutoExportEnabledEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_AUTO_EXPORT");
constexpr auto kRiskHistoryAutoExportIntervalEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_AUTO_EXPORT_INTERVAL_MINUTES");
constexpr auto kRiskHistoryAutoExportBasenameEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_AUTO_EXPORT_BASENAME");
constexpr auto kRiskHistoryAutoExportDirEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_AUTO_EXPORT_DIR");
constexpr auto kRiskHistoryAutoExportLocalTimeEnv = QByteArrayLiteral("BOT_CORE_UI_RISK_HISTORY_AUTO_EXPORT_USE_LOCAL_TIME");

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

QString readTokenFile(const QString& rawPath, const QString& label = QStringLiteral("MetricsService"))
{
    if (rawPath.trimmed().isEmpty())
        return {};
    const QString path = expandUserPath(rawPath.trimmed());
    QFile file(path);
    if (!file.exists()) {
        qCWarning(lcAppMetrics) << "Plik z tokenem" << label << "nie istnieje:" << path;
        return {};
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCWarning(lcAppMetrics) << "Nie udało się odczytać pliku z tokenem" << label << ':' << path
                                << file.errorString();
        return {};
    }
    const QByteArray data = file.readAll();
    QString token = QString::fromUtf8(data).trimmed();
    if (token.isEmpty()) {
        qCWarning(lcAppMetrics) << "Plik" << path << "nie zawiera tokenu autoryzacyjnego" << label;
        return {};
    }
    return token;
}

QStringList splitScopesList(const QString& raw)
{
    QString normalized = raw;
    normalized.replace(QLatin1Char(';'), QLatin1Char(','));
    QStringList parts = normalized.split(QRegularExpression(QStringLiteral("[\n\r\t, ]+")),
                                        Qt::SkipEmptyParts);
    QStringList unique;
    QSet<QString> seen;
    for (QString part : parts) {
        const QString trimmed = part.trimmed();
        if (trimmed.isEmpty())
            continue;
        if (seen.contains(trimmed))
            continue;
        seen.insert(trimmed);
        unique.append(trimmed);
    }
    return unique;
}

QString sanitizeAutoExportBasename(const QString& raw)
{
    QString sanitized = raw.trimmed();
    if (sanitized.isEmpty())
        return QStringLiteral("risk-history");

    static const QRegularExpression invalidCharacters(QStringLiteral("[^A-Za-z0-9_-]+"));
    sanitized.replace(invalidCharacters, QStringLiteral("_"));

    static const QRegularExpression repeatedUnderscore(QStringLiteral("_+"));
    sanitized.replace(repeatedUnderscore, QStringLiteral("_"));

    while (sanitized.startsWith(QLatin1Char('_')))
        sanitized.remove(0, 1);
    while (sanitized.endsWith(QLatin1Char('_')))
        sanitized.chop(1);

    if (sanitized.isEmpty())
        return QStringLiteral("risk-history");

    constexpr int kMaxLength = 80;
    if (sanitized.size() > kMaxLength)
        sanitized = sanitized.left(kMaxLength);

    return sanitized;
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
    const QString configDir = QDir::current().absoluteFilePath(QStringLiteral("config"));
    const QString defaultLicensePath = QDir::current().absoluteFilePath(QStringLiteral("var/licenses/active/license.json"));
    m_licenseController->setConfigDirectory(configDir);
    m_licenseController->setLicenseStoragePath(defaultLicensePath);

    m_securityController = std::make_unique<SecurityAdminController>(this);
    m_securityController->setProfilesPath(QDir::current().absoluteFilePath(QStringLiteral("config/user_profiles.json")));
    m_securityController->setLicensePath(defaultLicensePath);
    m_securityController->setLogPath(QDir::current().absoluteFilePath(QStringLiteral("logs/security_admin.log")));

    m_reportController = std::make_unique<ReportCenterController>(this);
    m_reportController->setReportsDirectory(QDir::current().absoluteFilePath(QStringLiteral("var/reports")));
    m_reportController->setReportsRoot(QDir::current().absoluteFilePath(QStringLiteral("var/reports")));

    m_strategyController = std::make_unique<StrategyConfigController>(this);
    m_strategyController->setConfigPath(QDir::current().absoluteFilePath(QStringLiteral("config/core.yaml")));
    m_strategyController->setScriptPath(QDir::current().absoluteFilePath(QStringLiteral("scripts/ui_config_bridge.py")));

    m_supportController = std::make_unique<SupportBundleController>(this);

    m_filteredAlertsModel.setSourceModel(&m_alertsModel);
    m_filteredAlertsModel.setSeverityFilter(AlertsFilterProxyModel::WarningsAndCritical);

    connect(&m_filteredAlertsModel, &AlertsFilterProxyModel::filterChanged, this, [this]() {
        if (!m_loadingUiSettings)
            scheduleUiSettingsPersist();
    });

    initializeUiSettingsStorage();

    m_repoRoot = locateRepoRoot();

    if (m_supportController) {
        if (!m_repoRoot.isEmpty())
            m_supportController->setScriptPath(QDir(m_repoRoot).absoluteFilePath(QStringLiteral("scripts/export_support_bundle.py")));
        else
            m_supportController->setScriptPath(QDir::current().absoluteFilePath(QStringLiteral("scripts/export_support_bundle.py")));
    }

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

    connect(&m_alertsModel, &AlertsModel::acknowledgementsChanged, this, [this]() {
        if (!m_loadingUiSettings)
            scheduleUiSettingsPersist();
    });

    connect(&m_riskHistoryModel, &RiskHistoryModel::historyChanged, this, [this]() {
        if (!m_loadingUiSettings)
            scheduleUiSettingsPersist();
    });

    connect(&m_riskHistoryModel, &RiskHistoryModel::maximumEntriesChanged, this, [this]() {
        if (!m_loadingUiSettings)
            scheduleUiSettingsPersist();
    });

    connect(&m_riskHistoryModel, &RiskHistoryModel::snapshotRecorded, this,
            &Application::handleRiskHistorySnapshotRecorded);

    m_riskRefreshTimer.setTimerType(Qt::VeryCoarseTimer);
    m_riskRefreshTimer.setInterval(m_riskRefreshIntervalMs);
    m_riskRefreshTimer.setSingleShot(false);
    m_riskRefreshTimer.setParent(this);
    connect(&m_riskRefreshTimer, &QTimer::timeout, this, [this]() {
        if (!m_riskRefreshEnabled)
            return;

        m_lastRiskRefreshRequestUtc = QDateTime::currentDateTimeUtc();
        m_client.refreshRiskState();
        m_nextRiskRefreshUtc = m_lastRiskRefreshRequestUtc.addMSecs(m_riskRefreshIntervalMs);
        Q_EMIT riskRefreshScheduleChanged();
    });
}

Application::~Application()
{
    if (m_localService)
        m_localService->stop();
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
    parser.addOption({"risk-refresh-interval",
                      tr("Interwał automatycznego odświeżania ryzyka (s)"), tr("seconds"),
                      QString::number(kDefaultRiskRefreshSeconds, 'f', 1)});
    parser.addOption({"risk-refresh-disable",
                      tr("Wyłącz automatyczne pobieranie stanu ryzyka")});
    parser.addOption({"risk-history-export-dir",
                      tr("Katalog eksportu historii ryzyka"), tr("path"), QString()});
    parser.addOption({"risk-history-export-limit",
                      tr("Limit próbek eksportowanych do CSV"), tr("count"), QString()});
    parser.addOption({"risk-history-export-limit-disable",
                      tr("Wyłącza limit próbek eksportu historii ryzyka")});
    parser.addOption({"risk-history-auto-export",
                      tr("Włącza automatyczny eksport historii ryzyka")});
    parser.addOption({"risk-history-auto-export-disable",
                      tr("Wyłącza automatyczny eksport historii ryzyka")});
    parser.addOption({"risk-history-auto-export-interval",
                      tr("Interwał autoeksportu historii ryzyka (min)"), tr("minutes"), QString()});
    parser.addOption({"risk-history-auto-export-basename",
                      tr("Prefiks plików autoeksportu historii ryzyka"), tr("name"), QString()});
    parser.addOption({"risk-history-auto-export-local-time",
                      tr("Nazwy plików autoeksportu używają czasu lokalnego")});
    parser.addOption({"risk-history-auto-export-utc",
                      tr("Nazwy plików autoeksportu używają czasu UTC")});
    parser.addOption({"risk-history-auto-export-dir",
                      tr("Katalog docelowy automatycznego eksportu historii ryzyka"), tr("path"), QString()});
    parser.addOption({"ui-settings-path", tr("Ścieżka pliku ustawień UI"), tr("path"), QString()});
    parser.addOption({"disable-ui-settings", tr("Wyłącza zapisywanie konfiguracji UI")});
    parser.addOption({"enable-ui-settings",
                      tr("Wymusza zapisywanie konfiguracji UI nawet przy dezaktywacji w zmiennych środowiskowych")});

    parser.addOption({"screen-name", tr("Preferowany ekran (nazwa QScreen)"), tr("name")});
    parser.addOption({"screen-index", tr("Preferowany ekran (indeks)"), tr("index")});
    parser.addOption({"primary-screen", tr("Wymusza użycie ekranu podstawowego")});

    // Telemetria MetricsService
    parser.addOption({"metrics-endpoint", tr("Adres serwera MetricsService"), tr("endpoint"),
                      QStringLiteral("127.0.0.1:50061")});
    parser.addOption({"metrics-tag", tr("Etykieta notatek telemetrii"), tr("tag"), QString()});
    parser.addOption({"metrics-auth-token", tr("Token autoryzacyjny MetricsService"), tr("token")});
    parser.addOption({"metrics-auth-token-file", tr("Ścieżka pliku z tokenem MetricsService"), tr("path")});
    parser.addOption({"metrics-rbac-role", tr("Rola RBAC przekazywana do MetricsService"), tr("role"), QString()});
    parser.addOption({"disable-metrics", tr("Wyłącza wysyłkę telemetrii")});
    parser.addOption({"no-metrics", tr("Alias: wyłącza wysyłkę telemetrii")});

    // Lokalny stub bot_core (gRPC)
    parser.addOption({"local-core", tr("Uruchamia lokalny stub API bot_core")});
    parser.addOption({"no-local-core", tr("Wyłącza lokalny stub API bot_core")});
    parser.addOption({"local-core-host", tr("Adres nasłuchu stubu"), tr("host"), QStringLiteral("127.0.0.1")});
    parser.addOption({"local-core-port", tr("Port stubu (0 = przydziel losowo)"), tr("port"), QStringLiteral("0")});
    parser.addOption({"local-core-python", tr("Interpreter Pythona stubu bot_core"), tr("path"), QString()});
    parser.addOption({"local-core-dataset", tr("Plik YAML z danymi stubu"), tr("path"), QString()});
    parser.addOption({"local-core-stream-repeat", tr("Powtarza strumień OHLCV stubu")});
    parser.addOption({"local-core-stream-interval", tr("Opóźnienie strumienia (s)"), tr("seconds"), QStringLiteral("0")});

    // TLS/mTLS gRPC (demon tradingowy)
    parser.addOption({"grpc-use-mtls", tr("Wymusza mTLS dla klienta tradingowego")});
    parser.addOption({"grpc-root-cert", tr("Root CA (PEM) dla kanału tradingowego"), tr("path"), QString()});
    parser.addOption({"grpc-client-cert", tr("Certyfikat klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"grpc-client-key", tr("Klucz klienta (PEM)"), tr("path"), QString()});
    parser.addOption({"grpc-auth-token", tr("Token autoryzacyjny TradingService"), tr("token")});
    parser.addOption({"grpc-auth-token-file", tr("Ścieżka pliku z tokenem TradingService"), tr("path")});
    parser.addOption({"grpc-rbac-role", tr("Rola RBAC przekazywana do TradingService"), tr("role"), QString()});
    parser.addOption({"grpc-rbac-scopes", tr("Lista scope RBAC (oddzielone przecinkami)") , tr("scopes"), QString()});
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
    parser.addOption({"security-profiles-path", tr("Ścieżka profili użytkowników UI"), tr("path"), QString()});
    parser.addOption({"security-python", tr("Ścieżka do interpretera Pythona dla bridge"), tr("path"), QString()});
    parser.addOption({"security-log-path", tr("Plik logu zdarzeń administracyjnych"), tr("path"), QString()});
    parser.addOption({"reports-directory", tr("Katalog raportów pipeline"), tr("path"),
                      QStringLiteral("var/reports")});
    parser.addOption({"reporting-python", tr("Interpreter Pythona mostka raportów"), tr("path"), QString()});
    parser.addOption({"core-config", tr("Plik głównej konfiguracji core.yaml"), tr("path"),
                      QStringLiteral("config/core.yaml")});
    parser.addOption({"strategy-config-python", tr("Interpreter Pythona mostka konfiguracji strategii"), tr("path"),
                      QString()});
    parser.addOption({"strategy-config-bridge", tr("Ścieżka do scripts/ui_config_bridge.py"), tr("path"), QString()});
    parser.addOption({"support-bundle-python", tr("Interpreter Pythona eksportu pakietu wsparcia"), tr("path"), QString()});
    parser.addOption({"support-bundle-script", tr("Ścieżka do scripts/export_support_bundle.py"), tr("path"), QString()});
    parser.addOption({"support-bundle-output-dir", tr("Katalog docelowy pakietów wsparcia"), tr("path"), QString()});
    parser.addOption({"support-bundle-format", tr("Format pakietu wsparcia (tar.gz lub zip)"), tr("format"),
                      QStringLiteral("tar.gz")});
    parser.addOption({"support-bundle-basename", tr("Bazowa nazwa pliku pakietu wsparcia"), tr("name"),
                      QStringLiteral("support-bundle")});
    parser.addOption({"support-bundle-include", tr("Dodatkowa ścieżka pakietu wsparcia (label=path)"), tr("spec")});
    parser.addOption({"support-bundle-disable", tr("Wyłącz domyślny zasób pakietu wsparcia (np. logs)"), tr("label")});
    parser.addOption({"support-bundle-metadata", tr("Para metadata key=value dla pakietu wsparcia"), tr("pair")});
}

bool Application::applyParser(const QCommandLineParser& parser) {
    TradingClient::InstrumentConfig instrument;
    instrument.exchange = parser.value("exchange");
    instrument.symbol = parser.value("symbol");
    instrument.venueSymbol = parser.value("venue-symbol");
    instrument.quoteCurrency = parser.value("quote");
    instrument.baseCurrency = parser.value("base");
    instrument.granularityIso8601 = parser.value("granularity");

    QString endpoint = parser.value("endpoint");
    configureLocalBotCoreService(parser, endpoint);
    m_client.setEndpoint(endpoint);
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
    if (m_localServiceEnabled) {
        tradingTls.enabled = false;
        tradingTls.rootCertificatePath.clear();
        tradingTls.clientCertificatePath.clear();
        tradingTls.clientKeyPath.clear();
        tradingTls.targetNameOverride.clear();
    }
    m_tradingTlsConfig = tradingTls;

    QString cliTradingToken = parser.value("grpc-auth-token").trimmed();
    QString cliTradingTokenFile = parser.value("grpc-auth-token-file").trimmed();
    const bool cliTradingTokenProvided = !cliTradingToken.isEmpty();
    const bool cliTradingTokenFileProvided = !cliTradingTokenFile.isEmpty();
    if (cliTradingTokenProvided && cliTradingTokenFileProvided) {
        qCWarning(lcAppMetrics)
            << "Podano jednocześnie --grpc-auth-token oraz --grpc-auth-token-file. Użyję tokenu przekazanego bezpośrednio.";
    }

    if (cliTradingTokenProvided) {
        m_tradingAuthToken = cliTradingToken;
    } else if (cliTradingTokenFileProvided) {
        m_tradingAuthToken = readTokenFile(cliTradingTokenFile, QStringLiteral("TradingService"));
    } else {
        m_tradingAuthToken.clear();
    }

    QString cliTradingRole = parser.value("grpc-rbac-role").trimmed();
    const bool cliTradingRoleProvided = !cliTradingRole.isEmpty();
    if (cliTradingRoleProvided) {
        m_tradingRbacRole = cliTradingRole;
    } else {
        m_tradingRbacRole.clear();
    }

    QString cliTradingScopesRaw = parser.value("grpc-rbac-scopes").trimmed();
    const bool cliTradingScopesProvided = !cliTradingScopesRaw.isEmpty();
    if (cliTradingScopesProvided) {
        m_tradingRbacScopes = splitScopesList(cliTradingScopesRaw);
    } else {
        m_tradingRbacScopes.clear();
    }

    // --- Telemetria ---
    m_metricsEndpoint = parser.value("metrics-endpoint");
    if (m_metricsEndpoint.isEmpty()) {
        // Fallback: użyj endpointu tradingowego, jeśli nie podano dedykowanego dla MetricsService
        m_metricsEndpoint = endpoint;
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

    m_metricsRbacRole = parser.value("metrics-rbac-role").trimmed();

    applyTradingTlsEnvironmentOverrides(parser);
    applyTradingAuthEnvironmentOverrides(parser,
                                         cliTradingTokenProvided,
                                         cliTradingTokenFileProvided,
                                         cliTradingRoleProvided,
                                         cliTradingScopesProvided);
    m_client.setTlsConfig(m_tradingTlsConfig);
    m_client.setAuthToken(m_tradingAuthToken);
    m_client.setRbacRole(m_tradingRbacRole);
    m_client.setRbacScopes(m_tradingRbacScopes);

    bool riskRefreshEnabled = !parser.isSet("risk-refresh-disable");
    bool intervalOk = false;
    double riskRefreshSeconds = parser.value("risk-refresh-interval").toDouble(&intervalOk);
    if (!intervalOk || riskRefreshSeconds <= 0.0) {
        if (parser.isSet("risk-refresh-interval")) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowy interwał --risk-refresh-interval:" << parser.value("risk-refresh-interval")
                << "– używam wartości domyślnej";
        }
        riskRefreshSeconds = kDefaultRiskRefreshSeconds;
    }

    if (!parser.isSet("risk-refresh-interval")) {
        if (const auto envInterval = envValue("BOT_CORE_UI_RISK_REFRESH_SECONDS")) {
            bool envOk = false;
            const double candidate = envInterval->toDouble(&envOk);
            if (envOk && candidate > 0.0) {
                riskRefreshSeconds = candidate;
            } else {
                qCWarning(lcAppMetrics)
                    << "Nieprawidłowa wartość BOT_CORE_UI_RISK_REFRESH_SECONDS:" << *envInterval
                    << "– oczekiwano liczby dodatniej";
            }
        }
    }

    if (!parser.isSet("risk-refresh-disable")) {
        if (const auto envDisable = envBool("BOT_CORE_UI_RISK_REFRESH_DISABLE"))
            riskRefreshEnabled = !envDisable.value();
    }

    configureRiskRefresh(riskRefreshEnabled, riskRefreshSeconds);

    applyRiskHistoryCliOverrides(parser);
    configureStrategyBridge(parser);
    configureSupportBundle(parser);

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

    if (m_securityController) {
        if (!parser.value("security-profiles-path").trimmed().isEmpty()) {
            m_securityController->setProfilesPath(expandUserPath(parser.value("security-profiles-path")));
        }
        if (!parser.value("security-python").trimmed().isEmpty()) {
            m_securityController->setPythonExecutable(parser.value("security-python"));
        }
        if (!parser.value("security-log-path").trimmed().isEmpty()) {
            m_securityController->setLogPath(expandUserPath(parser.value("security-log-path")));
        }
        m_securityController->refresh();
    }

    if (m_reportController) {
        const auto applyReportsDirectory = [this](const QString& candidate) {
            const QString trimmed = candidate.trimmed();
            if (trimmed.isEmpty())
                return;
            m_reportController->setReportsDirectory(expandUserPath(trimmed));
        };
        if (parser.isSet("reports-directory")) {
            applyReportsDirectory(parser.value("reports-directory"));
        } else {
            const auto envReports = envValue("BOT_CORE_UI_REPORTS_DIR");
            if (envReports.has_value()) {
                const QString envReportsValue = envReports->trimmed();
                if (!envReportsValue.isEmpty()) {
                    applyReportsDirectory(envReportsValue);
                } else {
                    applyReportsDirectory(parser.value("reports-directory"));
                }
            } else {
                applyReportsDirectory(parser.value("reports-directory"));
            }
        }

        const auto applyPythonExecutable = [this](const QString& candidate) {
            const QString trimmed = candidate.trimmed();
            if (trimmed.isEmpty())
                return;
            m_reportController->setPythonExecutable(expandUserPath(trimmed));
        };
        if (parser.isSet("reporting-python")) {
            applyPythonExecutable(parser.value("reporting-python"));
        } else {
            const auto envPython = envValue("BOT_CORE_UI_REPORTS_PYTHON");
            if (envPython.has_value()) {
                const QString envPythonValue = envPython->trimmed();
                if (!envPythonValue.isEmpty())
                    applyPythonExecutable(envPythonValue);
            }
        }

        m_reportController->refresh();
    }

    applyUiSettingsCliOverrides(parser);

    // Inicjalizacja/reportera + token
    ensureTelemetry();

    return true;
}

QString Application::locateRepoRoot() const
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

void Application::configureRiskRefresh(bool enabled, double intervalSeconds)
{
    double sanitizedSeconds = intervalSeconds > 0.0 ? intervalSeconds : kDefaultRiskRefreshSeconds;
    const int rawMs = static_cast<int>(std::llround(sanitizedSeconds * 1000.0));
    const int clampedMs = qBound(kMinRiskRefreshIntervalMs, rawMs, kMaxRiskRefreshIntervalMs);

    m_riskRefreshIntervalMs = clampedMs;
    m_riskRefreshEnabled = enabled && sanitizedSeconds > 0.0;

    m_riskRefreshTimer.stop();
    m_riskRefreshTimer.setInterval(m_riskRefreshIntervalMs);
    if (!m_riskRefreshEnabled)
        m_nextRiskRefreshUtc = {};
    Q_EMIT riskRefreshScheduleChanged();
}

void Application::applyRiskRefreshTimerState()
{
    if (m_riskRefreshEnabled && m_started) {
        const bool needsRestart = !m_riskRefreshTimer.isActive()
            || m_riskRefreshTimer.interval() != m_riskRefreshIntervalMs;
        if (needsRestart)
            m_riskRefreshTimer.start(m_riskRefreshIntervalMs);
        m_nextRiskRefreshUtc = QDateTime::currentDateTimeUtc().addMSecs(m_riskRefreshIntervalMs);
    } else {
        if (m_riskRefreshTimer.isActive())
            m_riskRefreshTimer.stop();
        m_nextRiskRefreshUtc = {};
    }
    Q_EMIT riskRefreshScheduleChanged();
}

void Application::initializeUiSettingsStorage()
{
    if (const auto disableEnv = envBool(kUiSettingsDisableEnv); disableEnv.has_value())
        m_uiSettingsPersistenceEnabled = !disableEnv.value();

    if (m_uiSettingsPath.trimmed().isEmpty()) {
        QString candidate;
        if (const auto envPath = envValue(kUiSettingsEnv); envPath.has_value()) {
            const QString trimmed = envPath->trimmed();
            if (!trimmed.isEmpty())
                candidate = expandUserPath(trimmed);
        }

        if (candidate.isEmpty())
            candidate = QDir::current().absoluteFilePath(QStringLiteral("var/state/ui_settings.json"));

        QFileInfo info(candidate);
        if (!info.isAbsolute())
            candidate = QDir::current().absoluteFilePath(candidate);

        m_uiSettingsPath = candidate;
    } else {
        QFileInfo info(m_uiSettingsPath);
        if (!info.isAbsolute())
            m_uiSettingsPath = QDir::current().absoluteFilePath(m_uiSettingsPath);
    }

    ensureUiSettingsTimerConfigured();

    if (!m_uiSettingsPersistenceEnabled)
        return;

    loadUiSettings();
}

void Application::ensureUiSettingsTimerConfigured()
{
    if (m_uiSettingsTimerConfigured)
        return;

    m_uiSettingsSaveTimer.setParent(this);
    m_uiSettingsSaveTimer.setSingleShot(true);
    m_uiSettingsSaveTimer.setInterval(kUiSettingsDebounceMs);
    connect(&m_uiSettingsSaveTimer, &QTimer::timeout, this, &Application::persistUiSettings);
    m_uiSettingsTimerConfigured = true;
}

void Application::applyUiSettingsCliOverrides(const QCommandLineParser& parser)
{
    if (parser.isSet("enable-ui-settings"))
        setUiSettingsPersistenceEnabled(true);
    if (parser.isSet("disable-ui-settings"))
        setUiSettingsPersistenceEnabled(false);

    const QString cliPath = parser.value("ui-settings-path").trimmed();
    if (!cliPath.isEmpty())
        setUiSettingsPath(cliPath);
}

void Application::applyRiskHistoryCliOverrides(const QCommandLineParser& parser)
{
    const bool previousLoading = m_loadingUiSettings;
    m_loadingUiSettings = true;

    const auto restoreLoadingFlag = qScopeGuard([this, previousLoading]() {
        m_loadingUiSettings = previousLoading;
    });

    const auto applyDirectory = [this](const QString& raw) {
        const QString trimmed = raw.trimmed();
        if (trimmed.isEmpty())
            return;

        QUrl url(trimmed);
        if (!url.isValid() || url.scheme().isEmpty()) {
            const QString expanded = expandUserPath(trimmed);
            const QString absolute = QDir(expanded).absolutePath();
            url = QUrl::fromLocalFile(absolute);
        } else if (url.isLocalFile()) {
            url = QUrl::fromLocalFile(QDir(url.toLocalFile()).absolutePath());
        }

        if (!url.isValid() || (!url.isLocalFile() && !url.scheme().isEmpty())) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowy katalog eksportu historii ryzyka:" << raw;
            return;
        }

        setRiskHistoryExportLastDirectory(url);
    };

    const auto applyLimitValue = [this](const QString& raw) -> bool {
        const QString trimmed = raw.trimmed();
        if (trimmed.isEmpty())
            return false;

        bool ok = false;
        const int value = trimmed.toInt(&ok);
        if (!ok) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowy limit eksportu historii ryzyka:" << raw;
            return false;
        }

        if (value <= 0) {
            setRiskHistoryExportLimitEnabled(false);
            return true;
        }

        setRiskHistoryExportLimitValue(value);
        setRiskHistoryExportLimitEnabled(true);
        return true;
    };

    const auto applyAutoInterval = [this](const QString& raw) {
        const QString trimmed = raw.trimmed();
        if (trimmed.isEmpty())
            return;

        bool ok = false;
        int minutes = trimmed.toInt(&ok);
        if (!ok) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowy interwał autoeksportu historii ryzyka:" << raw;
            return;
        }

        minutes = qMax(1, minutes);
        setRiskHistoryAutoExportIntervalMinutes(minutes);
    };

    const auto applyBasename = [this](const QString& raw) {
        const QString trimmed = raw.trimmed();
        if (trimmed.isEmpty())
            return;
        setRiskHistoryAutoExportBasename(trimmed);
    };

    if (parser.isSet("risk-history-export-dir"))
        applyDirectory(parser.value("risk-history-export-dir"));
    else if (const auto envDir = envValue(kRiskHistoryExportDirEnv); envDir.has_value())
        applyDirectory(envDir->trimmed());

    if (parser.isSet("risk-history-auto-export-dir"))
        applyDirectory(parser.value("risk-history-auto-export-dir"));
    else if (const auto envAutoDir = envValue(kRiskHistoryAutoExportDirEnv); envAutoDir.has_value())
        applyDirectory(envAutoDir->trimmed());

    bool limitValueApplied = false;
    bool limitEnabledForced = false;

    if (parser.isSet("risk-history-export-limit")) {
        limitValueApplied = applyLimitValue(parser.value("risk-history-export-limit"));
        if (limitValueApplied)
            limitEnabledForced = true;
    }

    if (parser.isSet("risk-history-export-limit-disable")) {
        setRiskHistoryExportLimitEnabled(false);
        limitEnabledForced = true;
    }

    if (!limitValueApplied) {
        if (const auto envLimit = envValue(kRiskHistoryExportLimitEnv); envLimit.has_value())
            limitValueApplied = applyLimitValue(envLimit->trimmed());
    }

    if (!limitEnabledForced) {
        if (const auto envLimitEnabled = envBool(kRiskHistoryExportLimitEnabledEnv); envLimitEnabled.has_value()) {
            setRiskHistoryExportLimitEnabled(envLimitEnabled.value());
            limitEnabledForced = true;
        }
    }

    bool autoExportEnabledForced = false;
    if (parser.isSet("risk-history-auto-export")) {
        setRiskHistoryAutoExportEnabled(true);
        autoExportEnabledForced = true;
    }
    if (parser.isSet("risk-history-auto-export-disable")) {
        setRiskHistoryAutoExportEnabled(false);
        autoExportEnabledForced = true;
    }
    if (!autoExportEnabledForced) {
        if (const auto envAutoEnabled = envBool(kRiskHistoryAutoExportEnabledEnv); envAutoEnabled.has_value()) {
            setRiskHistoryAutoExportEnabled(envAutoEnabled.value());
            autoExportEnabledForced = true;
        }
    }

    bool autoExportTimeForced = false;
    if (parser.isSet("risk-history-auto-export-local-time")) {
        setRiskHistoryAutoExportUseLocalTime(true);
        autoExportTimeForced = true;
    }
    if (parser.isSet("risk-history-auto-export-utc")) {
        setRiskHistoryAutoExportUseLocalTime(false);
        autoExportTimeForced = true;
    }
    if (!autoExportTimeForced) {
        if (const auto envLocalTime = envBool(kRiskHistoryAutoExportLocalTimeEnv); envLocalTime.has_value()) {
            setRiskHistoryAutoExportUseLocalTime(envLocalTime.value());
            autoExportTimeForced = true;
        }
    }

    bool intervalApplied = false;
    if (parser.isSet("risk-history-auto-export-interval")) {
        applyAutoInterval(parser.value("risk-history-auto-export-interval"));
        intervalApplied = true;
    }
    if (!intervalApplied) {
        if (const auto envInterval = envValue(kRiskHistoryAutoExportIntervalEnv); envInterval.has_value())
            applyAutoInterval(envInterval->trimmed());
    }

    if (parser.isSet("risk-history-auto-export-basename"))
        applyBasename(parser.value("risk-history-auto-export-basename"));
    else if (const auto envBasename = envValue(kRiskHistoryAutoExportBasenameEnv); envBasename.has_value())
        applyBasename(envBasename->trimmed());
}

void Application::configureStrategyBridge(const QCommandLineParser& parser)
{
    if (!m_strategyController)
        return;

    QString configPath = parser.value("core-config").trimmed();
    if (!parser.isSet("core-config")) {
        if (const auto envConfig = envValue(QByteArrayLiteral("BOT_CORE_UI_CORE_CONFIG_PATH")))
            configPath = envConfig->trimmed();
    }
    if (configPath.isEmpty())
        configPath = QStringLiteral("config/core.yaml");
    m_strategyController->setConfigPath(expandUserPath(configPath));

    QString pythonExec = parser.value("strategy-config-python").trimmed();
    if (pythonExec.isEmpty()) {
        if (const auto envPython = envValue(QByteArrayLiteral("BOT_CORE_UI_STRATEGY_PYTHON")))
            pythonExec = envPython->trimmed();
    }
    if (!pythonExec.isEmpty())
        m_strategyController->setPythonExecutable(expandUserPath(pythonExec));

    QString bridgePath = parser.value("strategy-config-bridge").trimmed();
    if (bridgePath.isEmpty()) {
        if (const auto envBridge = envValue(QByteArrayLiteral("BOT_CORE_UI_STRATEGY_BRIDGE")))
            bridgePath = envBridge->trimmed();
    }
    if (bridgePath.isEmpty()) {
        if (!m_repoRoot.isEmpty())
            bridgePath = QDir(m_repoRoot).absoluteFilePath(QStringLiteral("scripts/ui_config_bridge.py"));
        else
            bridgePath = QDir::current().absoluteFilePath(QStringLiteral("scripts/ui_config_bridge.py"));
    }
    m_strategyController->setScriptPath(expandUserPath(bridgePath));

    if (!m_strategyController->refresh()) {
        const QString error = m_strategyController->lastError();
        if (!error.isEmpty())
            qCWarning(lcAppMetrics) << "Mostek konfiguracji strategii zwrócił błąd:" << error;
    }
}

void Application::configureSupportBundle(const QCommandLineParser& parser)
{
    if (!m_supportController)
        return;

    const auto envTrimmed = [](const QByteArray& key) -> std::optional<QString> {
        if (const auto value = envValue(key); value.has_value())
            return value->trimmed();
        return std::nullopt;
    };

    if (parser.isSet("support-bundle-python"))
        m_supportController->setPythonExecutable(expandUserPath(parser.value("support-bundle-python")));
    else if (const auto envPython = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_PYTHON")); envPython.has_value())
        m_supportController->setPythonExecutable(expandUserPath(envPython.value()));

    if (parser.isSet("support-bundle-script"))
        m_supportController->setScriptPath(expandUserPath(parser.value("support-bundle-script")));
    else if (const auto envScript = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_SCRIPT")); envScript.has_value())
        m_supportController->setScriptPath(expandUserPath(envScript.value()));

    if (parser.isSet("support-bundle-output-dir"))
        m_supportController->setOutputDirectory(expandUserPath(parser.value("support-bundle-output-dir")));
    else if (const auto envOutput = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_OUTPUT_DIR")); envOutput.has_value())
        m_supportController->setOutputDirectory(expandUserPath(envOutput.value()));

    if (parser.isSet("support-bundle-format"))
        m_supportController->setFormat(parser.value("support-bundle-format"));
    else if (const auto envFormat = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_FORMAT")); envFormat.has_value())
        m_supportController->setFormat(envFormat.value());

    if (parser.isSet("support-bundle-basename"))
        m_supportController->setDefaultBasename(parser.value("support-bundle-basename"));
    else if (const auto envBasename = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_BASENAME")); envBasename.has_value())
        m_supportController->setDefaultBasename(envBasename.value());

    const auto splitSpecs = [](const QString& raw) {
        return raw.split(QRegularExpression(QStringLiteral("[;,\\n]")), Qt::SkipEmptyParts);
    };

    QStringList includeEnv;
    if (const auto envInclude = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_INCLUDE")); envInclude.has_value())
        includeEnv = splitSpecs(envInclude.value());
    const QStringList includeCli = parser.values(QStringLiteral("support-bundle-include"));

    QStringList disableEnv;
    if (const auto envDisable = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_DISABLE")); envDisable.has_value())
        disableEnv = splitSpecs(envDisable.value());
    const QStringList disableCli = parser.values(QStringLiteral("support-bundle-disable"));

    QStringList metadataEnv;
    if (const auto envMetadata = envTrimmed(QByteArrayLiteral("BOT_CORE_UI_SUPPORT_METADATA")); envMetadata.has_value())
        metadataEnv = splitSpecs(envMetadata.value());
    const QStringList metadataCli = parser.values(QStringLiteral("support-bundle-metadata"));

    QStringList extraOrder;
    QHash<QString, QPair<QString, QString>> extraMap;

    const auto storeExtra = [&](const QString& label, const QString& path) {
        const QString lower = label.toLower();
        if (!extraMap.contains(lower))
            extraOrder.append(lower);
        extraMap.insert(lower, qMakePair(label, path));
    };

    const auto handleInclude = [&](const QString& rawSpec) {
        const QString trimmed = rawSpec.trimmed();
        if (trimmed.isEmpty())
            return;
        const int eq = trimmed.indexOf('=');
        if (eq <= 0) {
            qCWarning(lcAppMetrics) << "Nieprawidłowa ścieżka pakietu wsparcia" << trimmed;
            return;
        }
        const QString label = trimmed.left(eq).trimmed();
        const QString path = trimmed.mid(eq + 1).trimmed();
        if (label.isEmpty() || path.isEmpty()) {
            qCWarning(lcAppMetrics) << "Nieprawidłowa ścieżka pakietu wsparcia" << trimmed;
            return;
        }
        const QString lower = label.toLower();
        const QString expandedPath = expandUserPath(path);
        if (lower == QStringLiteral("logs")) {
            m_supportController->setLogsPath(expandedPath);
            m_supportController->setIncludeLogs(true);
        } else if (lower == QStringLiteral("reports")) {
            m_supportController->setReportsPath(expandedPath);
            m_supportController->setIncludeReports(true);
        } else if (lower == QStringLiteral("licenses")) {
            m_supportController->setLicensesPath(expandedPath);
            m_supportController->setIncludeLicenses(true);
        } else if (lower == QStringLiteral("metrics")) {
            m_supportController->setMetricsPath(expandedPath);
            m_supportController->setIncludeMetrics(true);
        } else if (lower == QStringLiteral("audit")) {
            m_supportController->setAuditPath(expandedPath);
            m_supportController->setIncludeAudit(true);
        } else {
            storeExtra(label, expandedPath);
        }
    };

    for (const QString& spec : includeEnv)
        handleInclude(spec);
    for (const QString& spec : includeCli)
        handleInclude(spec);

    QSet<QString> disabledCustom;
    const auto handleDisable = [&](const QString& rawLabel) {
        const QString label = rawLabel.trimmed().toLower();
        if (label.isEmpty())
            return;
        if (label == QStringLiteral("logs")) {
            m_supportController->setIncludeLogs(false);
        } else if (label == QStringLiteral("reports")) {
            m_supportController->setIncludeReports(false);
        } else if (label == QStringLiteral("licenses")) {
            m_supportController->setIncludeLicenses(false);
        } else if (label == QStringLiteral("metrics")) {
            m_supportController->setIncludeMetrics(false);
        } else if (label == QStringLiteral("audit")) {
            m_supportController->setIncludeAudit(false);
        } else {
            disabledCustom.insert(label);
        }
    };

    for (const QString& spec : disableEnv)
        handleDisable(spec);
    for (const QString& spec : disableCli)
        handleDisable(spec);

    if (!disabledCustom.isEmpty()) {
        QStringList filteredOrder;
        QHash<QString, QPair<QString, QString>> filteredMap;
        for (const QString& key : std::as_const(extraOrder)) {
            if (disabledCustom.contains(key))
                continue;
            filteredOrder.append(key);
            filteredMap.insert(key, extraMap.value(key));
        }
        extraOrder = filteredOrder;
        extraMap = filteredMap;
    }

    QStringList extraSpecs;
    extraSpecs.reserve(extraOrder.size());
    for (const QString& key : std::as_const(extraOrder)) {
        const auto pair = extraMap.value(key);
        extraSpecs.append(QStringLiteral("%1=%2").arg(pair.first, pair.second));
    }
    m_supportController->setExtraIncludeSpecs(extraSpecs);

    QVariantMap metadata;
    metadata.insert(QStringLiteral("origin"), QStringLiteral("desktop_ui"));
    metadata.insert(QStringLiteral("instrument"), instrumentLabel());
    metadata.insert(QStringLiteral("exchange"), m_instrument.exchange);
    metadata.insert(QStringLiteral("symbol"), m_instrument.symbol);
    metadata.insert(QStringLiteral("connection_status"), m_connectionStatus);
    metadata.insert(QStringLiteral("app_version"), QCoreApplication::applicationVersion());
    metadata.insert(QStringLiteral("hostname"), QSysInfo::machineHostName());

    const auto applyMetadata = [&](const QString& rawSpec) {
        const QString trimmed = rawSpec.trimmed();
        if (trimmed.isEmpty())
            return;
        const int eq = trimmed.indexOf('=');
        if (eq <= 0) {
            qCWarning(lcAppMetrics) << "Nieprawidłowa para metadata pakietu wsparcia" << trimmed;
            return;
        }
        const QString key = trimmed.left(eq).trimmed();
        const QString value = trimmed.mid(eq + 1).trimmed();
        if (key.isEmpty())
            return;
        metadata.insert(key, value);
    };

    for (const QString& spec : metadataEnv)
        applyMetadata(spec);
    for (const QString& spec : metadataCli)
        applyMetadata(spec);

    m_supportController->setMetadata(metadata);
}

void Application::setUiSettingsPersistenceEnabled(bool enabled)
{
    if (m_uiSettingsPersistenceEnabled == enabled)
        return;

    m_uiSettingsPersistenceEnabled = enabled;
    if (!enabled) {
        if (m_uiSettingsSaveTimer.isActive())
            m_uiSettingsSaveTimer.stop();
        return;
    }

    ensureUiSettingsTimerConfigured();
    loadUiSettings();
}

void Application::setUiSettingsPath(const QString& path, bool reload)
{
    QString candidate = path.trimmed();
    if (candidate.isEmpty())
        return;

    candidate = expandUserPath(candidate);
    QFileInfo info(candidate);
    if (!info.isAbsolute())
        candidate = QDir::current().absoluteFilePath(candidate);

    if (m_uiSettingsPath == candidate)
        return;

    m_uiSettingsPath = candidate;

    if (m_uiSettingsPersistenceEnabled && reload)
        loadUiSettings();
}

void Application::loadUiSettings()
{
    if (!m_uiSettingsPersistenceEnabled)
        return;
    if (m_uiSettingsPath.trimmed().isEmpty())
        return;

    QFile file(m_uiSettingsPath);
    if (!file.exists())
        return;

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCWarning(lcAppMetrics) << "Nie udało się odczytać pliku ustawień UI" << m_uiSettingsPath
                                << file.errorString();
        return;
    }

    const QByteArray data = file.readAll();
    file.close();

    const QJsonDocument document = QJsonDocument::fromJson(data);
    if (!document.isObject()) {
        qCWarning(lcAppMetrics) << "Plik ustawień UI ma nieprawidłowy format JSON" << m_uiSettingsPath;
        return;
    }

    const QJsonObject root = document.object();

    const auto requireString = [](const QJsonObject& object, const QString& key) -> QString {
        const QJsonValue value = object.value(key);
        if (!value.isString())
            return {};
        return value.toString().trimmed();
    };

    m_loadingUiSettings = true;

    if (root.contains(QStringLiteral("instrument")) && root.value(QStringLiteral("instrument")).isObject()) {
        const QJsonObject instrument = root.value(QStringLiteral("instrument")).toObject();
        const QString exchange = requireString(instrument, QStringLiteral("exchange"));
        const QString symbol = requireString(instrument, QStringLiteral("symbol"));
        const QString venueSymbol = requireString(instrument, QStringLiteral("venueSymbol"));
        const QString quote = requireString(instrument, QStringLiteral("quoteCurrency"));
        const QString base = requireString(instrument, QStringLiteral("baseCurrency"));
        const QString granularity = requireString(instrument, QStringLiteral("granularity"));

        if (!exchange.isEmpty() && !symbol.isEmpty() && !venueSymbol.isEmpty() && !quote.isEmpty()
            && !base.isEmpty() && !granularity.isEmpty()) {
            updateInstrument(exchange, symbol, venueSymbol, quote, base, granularity);
        }
    }

    if (root.contains(QStringLiteral("performanceGuard"))
        && root.value(QStringLiteral("performanceGuard")).isObject()) {
        const QJsonObject guardObj = root.value(QStringLiteral("performanceGuard")).toObject();
        const int fpsTarget = guardObj.value(QStringLiteral("fpsTarget")).toInt(m_guard.fpsTarget);
        const double reduceMotionAfter = guardObj.value(QStringLiteral("reduceMotionAfter"))
                                             .toDouble(m_guard.reduceMotionAfterSeconds);
        const double jankThreshold = guardObj.value(QStringLiteral("jankThresholdMs"))
                                         .toDouble(m_guard.jankThresholdMs);
        const int overlays = guardObj.value(QStringLiteral("maxOverlayCount")).toInt(m_guard.maxOverlayCount);
        const int disableSecondary = guardObj.value(QStringLiteral("disableSecondaryWhenBelow"))
                                         .toInt(m_guard.disableSecondaryWhenFpsBelow);
        updatePerformanceGuard(fpsTarget, reduceMotionAfter, jankThreshold, overlays, disableSecondary);
    }

    if (root.contains(QStringLiteral("riskRefresh")) && root.value(QStringLiteral("riskRefresh")).isObject()) {
        const QJsonObject riskObj = root.value(QStringLiteral("riskRefresh")).toObject();
        const bool enabled = riskObj.value(QStringLiteral("enabled")).toBool(m_riskRefreshEnabled);
        const double intervalSeconds = riskObj.value(QStringLiteral("intervalSeconds"))
                                         .toDouble(static_cast<double>(m_riskRefreshIntervalMs) / 1000.0);
        updateRiskRefresh(enabled, intervalSeconds);
    }

    if (root.contains(QStringLiteral("alerts")) && root.value(QStringLiteral("alerts")).isObject()) {
        const QJsonObject alertsObj = root.value(QStringLiteral("alerts")).toObject();
        const bool hideAcknowledged = alertsObj.value(QStringLiteral("hideAcknowledged"))
                                          .toBool(m_filteredAlertsModel.hideAcknowledged());
        m_filteredAlertsModel.setHideAcknowledged(hideAcknowledged);

        const int severityValue = alertsObj.value(QStringLiteral("severityFilter"))
                                      .toInt(static_cast<int>(m_filteredAlertsModel.severityFilter()));
        if (severityValue >= AlertsFilterProxyModel::AllSeverities
            && severityValue <= AlertsFilterProxyModel::WarningOnly) {
            m_filteredAlertsModel.setSeverityFilter(
                static_cast<AlertsFilterProxyModel::SeverityFilter>(severityValue));
        }

        const int sortValue = alertsObj.value(QStringLiteral("sortMode"))
                                  .toInt(static_cast<int>(m_filteredAlertsModel.sortMode()));
        if (sortValue >= AlertsFilterProxyModel::NewestFirst
            && sortValue <= AlertsFilterProxyModel::TitleAscending) {
            m_filteredAlertsModel.setSortMode(static_cast<AlertsFilterProxyModel::SortMode>(sortValue));
        }

        if (alertsObj.contains(QStringLiteral("searchText")))
            m_filteredAlertsModel.setSearchText(alertsObj.value(QStringLiteral("searchText")).toString());

        const QJsonValue acknowledgedValue = alertsObj.value(QStringLiteral("acknowledgedIds"));
        if (acknowledgedValue.isArray()) {
            const QJsonArray ackArray = acknowledgedValue.toArray();
            QStringList ids;
            ids.reserve(ackArray.size());
            for (const QJsonValue& value : ackArray) {
                if (value.isString())
                    ids.append(value.toString());
            }
            m_alertsModel.setAcknowledgedAlertIds(ids);
        }
    }

    if (root.contains(QStringLiteral("riskHistory"))) {
        const QJsonValue riskHistoryValue = root.value(QStringLiteral("riskHistory"));
        if (riskHistoryValue.isArray()) {
            m_riskHistoryModel.restoreFromJson(riskHistoryValue.toArray());
        } else if (riskHistoryValue.isObject()) {
            const QJsonObject historyObject = riskHistoryValue.toObject();
            const QJsonValue maxValue = historyObject.value(QStringLiteral("maximumEntries"));
            if (maxValue.isDouble())
                m_riskHistoryModel.setMaximumEntries(maxValue.toInt(m_riskHistoryModel.maximumEntries()));

            const QJsonValue entriesValue = historyObject.value(QStringLiteral("entries"));
            if (entriesValue.isArray())
                m_riskHistoryModel.restoreFromJson(entriesValue.toArray());

            const QJsonValue exportValue = historyObject.value(QStringLiteral("export"));
            if (exportValue.isObject()) {
                const QJsonObject exportObject = exportValue.toObject();
                setRiskHistoryExportLimitEnabled(exportObject.value(QStringLiteral("limitEnabled"))
                                                    .toBool(m_riskHistoryExportLimitEnabled));

                const QJsonValue limitValue = exportObject.value(QStringLiteral("limitValue"));
                if (limitValue.isDouble())
                    setRiskHistoryExportLimitValue(std::max(1, limitValue.toInt(m_riskHistoryExportLimitValue)));

                if (exportObject.contains(QStringLiteral("lastDirectory"))) {
                    const QString lastDirValue = exportObject.value(QStringLiteral("lastDirectory"))
                                                       .toString();
                    const QString trimmed = lastDirValue.trimmed();
                    if (!trimmed.isEmpty()) {
                        QUrl directoryUrl(trimmed);
                        if (!directoryUrl.isValid() || directoryUrl.scheme().isEmpty())
                            directoryUrl = QUrl::fromLocalFile(expandUserPath(trimmed));
                        setRiskHistoryExportLastDirectory(directoryUrl);
                    }
                }

                const QJsonValue autoValue = exportObject.value(QStringLiteral("auto"));
                if (autoValue.isObject()) {
                    const QJsonObject autoObject = autoValue.toObject();
                    setRiskHistoryAutoExportEnabled(autoObject.value(QStringLiteral("enabled"))
                                                       .toBool(m_riskHistoryAutoExportEnabled));

                    const QJsonValue intervalValue = autoObject.value(QStringLiteral("intervalMinutes"));
                    if (intervalValue.isDouble()) {
                        const int intervalMinutes = std::max(1, intervalValue.toInt(m_riskHistoryAutoExportIntervalMinutes));
                        setRiskHistoryAutoExportIntervalMinutes(intervalMinutes);
                    }

                    if (autoObject.contains(QStringLiteral("basename")))
                        setRiskHistoryAutoExportBasename(autoObject.value(QStringLiteral("basename"))
                                                            .toString(m_riskHistoryAutoExportBasename));

                    if (autoObject.contains(QStringLiteral("useLocalTime")))
                        setRiskHistoryAutoExportUseLocalTime(autoObject.value(QStringLiteral("useLocalTime"))
                                                                .toBool(m_riskHistoryAutoExportUseLocalTime));

                    if (autoObject.contains(QStringLiteral("lastExportAt"))) {
                        const QString lastExportString = autoObject.value(QStringLiteral("lastExportAt")).toString();
                        QDateTime parsed = QDateTime::fromString(lastExportString, Qt::ISODateWithMs);
                        if (!parsed.isValid())
                            parsed = QDateTime::fromString(lastExportString, Qt::ISODate);
                        if (parsed.isValid()) {
                            parsed = parsed.toUTC();
                            m_lastRiskHistoryAutoExportUtc = parsed;
                            Q_EMIT riskHistoryLastAutoExportAtChanged();
                        }
                    }

                    if (autoObject.contains(QStringLiteral("lastPath"))) {
                        const QString lastPath = autoObject.value(QStringLiteral("lastPath")).toString();
                        if (!lastPath.trimmed().isEmpty()) {
                            QUrl pathUrl(lastPath);
                            if (!pathUrl.isValid() || pathUrl.scheme().isEmpty())
                                pathUrl = QUrl::fromLocalFile(expandUserPath(lastPath));
                            m_lastRiskHistoryAutoExportPath = pathUrl;
                            Q_EMIT riskHistoryLastAutoExportPathChanged();
                        }
                    }
                }
            }
        }
    }

    m_loadingUiSettings = false;
}

void Application::scheduleUiSettingsPersist()
{
    if (m_loadingUiSettings || !m_uiSettingsPersistenceEnabled || !m_uiSettingsTimerConfigured
        || m_uiSettingsPath.trimmed().isEmpty())
        return;

    m_uiSettingsSaveTimer.start();
}

void Application::persistUiSettings()
{
    if (!m_uiSettingsPersistenceEnabled || m_uiSettingsPath.trimmed().isEmpty())
        return;

    const QJsonObject payload = buildUiSettingsPayload();
    if (payload.isEmpty())
        return;

    const QFileInfo info(m_uiSettingsPath);
    QDir dir = info.dir();
    if (!dir.exists()) {
        if (!dir.mkpath(QStringLiteral("."))) {
            qCWarning(lcAppMetrics) << "Nie udało się utworzyć katalogu ustawień UI" << dir.absolutePath();
            return;
        }
    }

    QSaveFile file(m_uiSettingsPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qCWarning(lcAppMetrics) << "Nie udało się zapisać ustawień UI" << m_uiSettingsPath
                                << file.errorString();
        return;
    }

    const QJsonDocument document(payload);
    file.write(document.toJson(QJsonDocument::Compact));
    if (!file.commit()) {
        qCWarning(lcAppMetrics) << "Nie udało się zatwierdzić pliku ustawień UI" << m_uiSettingsPath
                                << file.errorString();
    } else {
        qCInfo(lcAppMetrics) << "Zapisano konfigurację UI do" << m_uiSettingsPath;
    }
}

QJsonObject Application::buildUiSettingsPayload() const
{
    QJsonObject root;

    QJsonObject instrument;
    instrument.insert(QStringLiteral("exchange"), m_instrument.exchange);
    instrument.insert(QStringLiteral("symbol"), m_instrument.symbol);
    instrument.insert(QStringLiteral("venueSymbol"), m_instrument.venueSymbol);
    instrument.insert(QStringLiteral("quoteCurrency"), m_instrument.quoteCurrency);
    instrument.insert(QStringLiteral("baseCurrency"), m_instrument.baseCurrency);
    instrument.insert(QStringLiteral("granularity"), m_instrument.granularityIso8601);
    root.insert(QStringLiteral("instrument"), instrument);

    QJsonObject guard;
    guard.insert(QStringLiteral("fpsTarget"), m_guard.fpsTarget);
    guard.insert(QStringLiteral("reduceMotionAfter"), m_guard.reduceMotionAfterSeconds);
    guard.insert(QStringLiteral("jankThresholdMs"), m_guard.jankThresholdMs);
    guard.insert(QStringLiteral("maxOverlayCount"), m_guard.maxOverlayCount);
    guard.insert(QStringLiteral("disableSecondaryWhenBelow"), m_guard.disableSecondaryWhenFpsBelow);
    root.insert(QStringLiteral("performanceGuard"), guard);

    QJsonObject risk;
    risk.insert(QStringLiteral("enabled"), m_riskRefreshEnabled);
    risk.insert(QStringLiteral("intervalSeconds"), static_cast<double>(m_riskRefreshIntervalMs) / 1000.0);
    root.insert(QStringLiteral("riskRefresh"), risk);

    QJsonObject alerts;
    QJsonArray acknowledged;
    const QStringList ackIds = m_alertsModel.acknowledgedAlertIds();
    for (const QString& id : ackIds)
        acknowledged.append(id);
    alerts.insert(QStringLiteral("acknowledgedIds"), acknowledged);
    alerts.insert(QStringLiteral("hideAcknowledged"), m_filteredAlertsModel.hideAcknowledged());
    alerts.insert(QStringLiteral("severityFilter"),
                  static_cast<int>(m_filteredAlertsModel.severityFilter()));
    alerts.insert(QStringLiteral("sortMode"), static_cast<int>(m_filteredAlertsModel.sortMode()));
    alerts.insert(QStringLiteral("searchText"), m_filteredAlertsModel.searchText());
    root.insert(QStringLiteral("alerts"), alerts);

    QJsonObject history;
    history.insert(QStringLiteral("maximumEntries"), m_riskHistoryModel.maximumEntries());
    const QJsonArray historyEntries = m_riskHistoryModel.toJson();
    if (!historyEntries.isEmpty())
        history.insert(QStringLiteral("entries"), historyEntries);

    QJsonObject exportPrefs;
    exportPrefs.insert(QStringLiteral("limitEnabled"), m_riskHistoryExportLimitEnabled);
    exportPrefs.insert(QStringLiteral("limitValue"), m_riskHistoryExportLimitValue);
    if (!m_riskHistoryExportLastDirectory.isEmpty())
        exportPrefs.insert(QStringLiteral("lastDirectory"),
                           m_riskHistoryExportLastDirectory.toString(QUrl::PreferLocalFile));
    QJsonObject autoPrefs;
    autoPrefs.insert(QStringLiteral("enabled"), m_riskHistoryAutoExportEnabled);
    autoPrefs.insert(QStringLiteral("intervalMinutes"), m_riskHistoryAutoExportIntervalMinutes);
    autoPrefs.insert(QStringLiteral("basename"), m_riskHistoryAutoExportBasename);
    autoPrefs.insert(QStringLiteral("useLocalTime"), m_riskHistoryAutoExportUseLocalTime);
    if (m_lastRiskHistoryAutoExportUtc.isValid())
        autoPrefs.insert(QStringLiteral("lastExportAt"),
                         m_lastRiskHistoryAutoExportUtc.toUTC().toString(Qt::ISODateWithMs));
    if (!m_lastRiskHistoryAutoExportPath.isEmpty())
        autoPrefs.insert(QStringLiteral("lastPath"),
                         m_lastRiskHistoryAutoExportPath.toString(QUrl::PreferLocalFile));
    exportPrefs.insert(QStringLiteral("auto"), autoPrefs);
    history.insert(QStringLiteral("export"), exportPrefs);
    root.insert(QStringLiteral("riskHistory"), history);

    return root;
}

void Application::maybeAutoExportRiskHistory(const QDateTime& snapshotTimestamp)
{
    if (!m_riskHistoryAutoExportEnabled)
        return;

    if (m_riskHistoryExportLastDirectory.isEmpty()) {
        if (!m_riskHistoryAutoExportDirectoryWarned) {
            qCWarning(lcAppMetrics)
                << "Automatyczny eksport historii ryzyka pominięty – brak skonfigurowanego katalogu docelowego.";
            m_riskHistoryAutoExportDirectoryWarned = true;
        }
        return;
    }

    QString directoryPath;
    if (m_riskHistoryExportLastDirectory.isLocalFile() || m_riskHistoryExportLastDirectory.scheme().isEmpty())
        directoryPath = m_riskHistoryExportLastDirectory.toLocalFile();
    else
        directoryPath = m_riskHistoryExportLastDirectory.toString(QUrl::PreferLocalFile);

    directoryPath = directoryPath.trimmed();
    if (directoryPath.isEmpty()) {
        if (!m_riskHistoryAutoExportDirectoryWarned) {
            qCWarning(lcAppMetrics)
                << "Automatyczny eksport historii ryzyka pominięty – ścieżka katalogu docelowego jest pusta.";
            m_riskHistoryAutoExportDirectoryWarned = true;
        }
        return;
    }

    QDir directory(directoryPath);
    if (!directory.exists() && !directory.mkpath(QStringLiteral("."))) {
        qCWarning(lcAppMetrics)
            << "Automatyczny eksport historii ryzyka pominięty – nie udało się utworzyć katalogu"
            << directory.absolutePath();
        return;
    }

    const QDateTime nowUtc = QDateTime::currentDateTimeUtc();
    const int intervalSeconds = qMax(1, m_riskHistoryAutoExportIntervalMinutes) * 60;
    if (m_lastRiskHistoryAutoExportUtc.isValid()) {
        if (m_lastRiskHistoryAutoExportUtc.secsTo(nowUtc) < intervalSeconds)
            return;
    }

    const QDateTime exportTimestamp = snapshotTimestamp.isValid() ? snapshotTimestamp : nowUtc;
    const QString filePath = resolveAutoExportFilePath(directory, m_riskHistoryAutoExportBasename, exportTimestamp);
    if (filePath.isEmpty()) {
        qCWarning(lcAppMetrics)
            << "Automatyczny eksport historii ryzyka pominięty – nie udało się wyznaczyć docelowej nazwy pliku.";
        return;
    }

    int limit = -1;
    if (m_riskHistoryExportLimitEnabled)
        limit = m_riskHistoryExportLimitValue;

    if (!m_riskHistoryModel.exportToCsv(filePath, limit)) {
        qCWarning(lcAppMetrics)
            << "Automatyczny eksport historii ryzyka nie powiódł się do pliku" << filePath;
        return;
    }

    m_lastRiskHistoryAutoExportUtc = nowUtc;
    m_lastRiskHistoryAutoExportPath = QUrl::fromLocalFile(filePath);
    m_riskHistoryAutoExportDirectoryWarned = false;
    Q_EMIT riskHistoryLastAutoExportAtChanged();
    Q_EMIT riskHistoryLastAutoExportPathChanged();
    qCInfo(lcAppMetrics) << "Automatycznie wyeksportowano historię ryzyka do" << filePath;
}

QString Application::resolveAutoExportFilePath(const QDir& directory,
                                               const QString& basename,
                                               const QDateTime& timestamp) const
{
    const QString sanitizedBase = sanitizeAutoExportBasename(basename);
    const QString effectiveBase = sanitizedBase.isEmpty() ? QStringLiteral("risk-history") : sanitizedBase;
    const bool useLocalTime = m_riskHistoryAutoExportUseLocalTime;

    QDateTime normalizedTimestamp;
    if (timestamp.isValid())
        normalizedTimestamp = useLocalTime ? timestamp.toLocalTime() : timestamp.toUTC();
    else
        normalizedTimestamp = useLocalTime ? QDateTime::currentDateTime() : QDateTime::currentDateTimeUtc();

    const QString timePart = normalizedTimestamp.toString(QStringLiteral("yyyyMMdd_HHmmss"));

    QString baseName;
    if (useLocalTime) {
        const int offsetSeconds = normalizedTimestamp.offsetFromUtc();
        const int offsetMinutes = offsetSeconds / 60;
        const int offsetHours = offsetMinutes / 60;
        const int remainingMinutes = std::abs(offsetMinutes % 60);
        const QChar sign = offsetSeconds >= 0 ? QLatin1Char('+') : QLatin1Char('-');
        const QString zoneSuffix = QStringLiteral("%1%2%3")
                                      .arg(sign)
                                      .arg(std::abs(offsetHours), 2, 10, QLatin1Char('0'))
                                      .arg(remainingMinutes, 2, 10, QLatin1Char('0'));
        baseName = QStringLiteral("%1_%2_%3").arg(effectiveBase, timePart, zoneSuffix);
    } else {
        baseName = QStringLiteral("%1_%2").arg(effectiveBase, timePart);
    }
    QString candidate = directory.absoluteFilePath(baseName + QStringLiteral(".csv"));
    if (!QFileInfo::exists(candidate))
        return candidate;

    for (int attempt = 1; attempt <= 100; ++attempt) {
        const QString alternate = directory.absoluteFilePath(
            QStringLiteral("%1_%2.csv").arg(baseName).arg(attempt));
        if (!QFileInfo::exists(alternate))
            return alternate;
    }

    return {};
}

void Application::configureLocalBotCoreService(const QCommandLineParser& parser, QString& endpoint)
{
    if (parser.isSet(QStringLiteral("no-local-core"))) {
        if (m_localService && m_localServiceEnabled) {
            m_localService->stop();
        }
        m_localServiceEnabled = false;
        return;
    }

    bool requested = true;
    if (const auto envFlag = envBool("BOT_CORE_UI_LOCAL_CORE")) {
        requested = envFlag.value();
    }
    if (parser.isSet(QStringLiteral("local-core"))) {
        requested = true;
    }

    if (!requested)
        return;

    if (!m_localService)
        m_localService = std::make_unique<BotCoreLocalService>(this);

    if (m_repoRoot.isEmpty())
        m_repoRoot = locateRepoRoot();

    if (!m_repoRoot.isEmpty())
        m_localService->setRepoRoot(m_repoRoot);

    QString pythonExecutable = parser.value(QStringLiteral("local-core-python")).trimmed();
    if (pythonExecutable.isEmpty()) {
        if (const auto envPython = envValue("BOT_CORE_UI_LOCAL_CORE_PYTHON"))
            pythonExecutable = envPython->trimmed();
    }
    if (!pythonExecutable.isEmpty())
        m_localService->setPythonExecutable(expandUserPath(pythonExecutable));

    QString datasetPath = parser.value(QStringLiteral("local-core-dataset")).trimmed();
    if (datasetPath.isEmpty()) {
        if (const auto envDataset = envValue("BOT_CORE_UI_LOCAL_CORE_DATASET"))
            datasetPath = envDataset->trimmed();
    }
    if (!datasetPath.isEmpty())
        m_localService->setDatasetPath(expandUserPath(datasetPath));

    QString host = parser.value(QStringLiteral("local-core-host")).trimmed();
    if (host.isEmpty()) {
        if (const auto envHost = envValue("BOT_CORE_UI_LOCAL_CORE_HOST"))
            host = envHost->trimmed();
    }
    if (host.isEmpty())
        host = QStringLiteral("127.0.0.1");
    m_localService->setHost(host);

    int port = 0;
    bool portOk = false;
    if (!parser.value(QStringLiteral("local-core-port")).trimmed().isEmpty()) {
        port = parser.value(QStringLiteral("local-core-port")).toInt(&portOk);
        if (!portOk) {
            qCWarning(lcAppMetrics)
                << "Nieprawidłowa wartość --local-core-port" << parser.value("local-core-port");
            port = 0;
        }
    }
    if (const auto envPort = envValue("BOT_CORE_UI_LOCAL_CORE_PORT")) {
        bool envOk = false;
        const int candidate = envPort->toInt(&envOk);
        if (envOk)
            port = candidate;
    }
    m_localService->setPort(port);

    bool repeat = parser.isSet(QStringLiteral("local-core-stream-repeat"));
    if (const auto envRepeat = envBool("BOT_CORE_UI_LOCAL_CORE_STREAM_REPEAT"))
        repeat = envRepeat.value();
    m_localService->setStreamRepeat(repeat);

    double interval = parser.value(QStringLiteral("local-core-stream-interval")).toDouble();
    if (const auto envInterval = envValue("BOT_CORE_UI_LOCAL_CORE_STREAM_INTERVAL")) {
        bool ok = false;
        const double candidate = envInterval->toDouble(&ok);
        if (ok)
            interval = candidate;
    }
    m_localService->setStreamInterval(interval);

    if (!m_localService->start()) {
        qCWarning(lcAppMetrics)
            << "Nie udało się uruchomić lokalnego serwisu bot_core:" << m_localService->lastError();
        m_localServiceEnabled = false;
        return;
    }

    endpoint = m_localService->endpoint();
    m_localServiceEnabled = true;
    qCInfo(lcAppMetrics) << "Uruchomiono lokalny stub bot_core pod adresem" << endpoint;
}

void Application::start() {
    m_ohlcvModel.setMaximumSamples(m_maxSamples);
    m_riskModel.clear();
    m_riskHistoryModel.clear();
    m_alertsModel.reset();

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
    m_started = true;
    applyRiskRefreshTimerState();
}

void Application::stop() {
    m_client.stop();
    m_alertsModel.reset();
    m_riskHistoryModel.clear();
    if (m_localService && m_localServiceEnabled) {
        m_localService->stop();
        m_localServiceEnabled = false;
    }
    m_started = false;
    applyRiskRefreshTimerState();
}

void Application::handleHistory(const QList<OhlcvPoint>& candles) {
    m_ohlcvModel.resetWithHistory(candles);
}

void Application::handleCandle(const OhlcvPoint& candle) {
    m_ohlcvModel.applyIncrement(candle);
}

void Application::handleRiskState(const RiskSnapshotData& snapshot) {
    m_riskModel.updateFromSnapshot(snapshot);
    m_riskHistoryModel.recordSnapshot(snapshot);
    m_alertsModel.updateFromRiskSnapshot(snapshot);
    if (snapshot.generatedAt.isValid())
        m_lastRiskUpdateUtc = snapshot.generatedAt.toUTC();
    else
        m_lastRiskUpdateUtc = QDateTime::currentDateTimeUtc();
    if (!m_lastRiskRefreshRequestUtc.isValid())
        m_lastRiskRefreshRequestUtc = m_lastRiskUpdateUtc;
    Q_EMIT riskRefreshScheduleChanged();
}

void Application::handleRiskHistorySnapshotRecorded(const QDateTime& timestamp)
{
    if (m_loadingUiSettings)
        return;

    maybeAutoExportRiskHistory(timestamp);
}

void Application::exposeToQml() {
    m_engine.rootContext()->setContextProperty(QStringLiteral("appController"), this);
    m_engine.rootContext()->setContextProperty(QStringLiteral("ohlcvModel"), &m_ohlcvModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("riskModel"), &m_riskModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("riskHistoryModel"), &m_riskHistoryModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("alertsModel"), &m_alertsModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("alertsFilterModel"), &m_filteredAlertsModel);
    m_engine.rootContext()->setContextProperty(QStringLiteral("licenseController"), m_licenseController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("activationController"), m_activationController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("securityController"), m_securityController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("reportController"), m_reportController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("strategyController"), m_strategyController.get());
    m_engine.rootContext()->setContextProperty(QStringLiteral("supportController"), m_supportController.get());
}

QObject* Application::activationController() const
{
    return m_activationController.get();
}

QObject* Application::reportController() const
{
    return m_reportController.get();
}

QObject* Application::strategyController() const
{
    return m_strategyController.get();
}

QObject* Application::supportController() const
{
    return m_supportController.get();
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
    if (auto* uiReporter = dynamic_cast<UiTelemetryReporter*>(m_telemetry.get())) {
        disconnect(uiReporter, &UiTelemetryReporter::pendingRetryCountChanged,
                   this, &Application::handleTelemetryPendingRetryCountChanged);
    }
    m_telemetry = std::move(reporter);
    updateTelemetryPendingRetryCount(0);
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

QVariantMap Application::instrumentConfigSnapshot() const {
    QVariantMap map;
    map.insert(QStringLiteral("exchange"), m_instrument.exchange);
    map.insert(QStringLiteral("symbol"), m_instrument.symbol);
    map.insert(QStringLiteral("venueSymbol"), m_instrument.venueSymbol);
    map.insert(QStringLiteral("quoteCurrency"), m_instrument.quoteCurrency);
    map.insert(QStringLiteral("baseCurrency"), m_instrument.baseCurrency);
    map.insert(QStringLiteral("granularity"), m_instrument.granularityIso8601);
    return map;
}

QVariantMap Application::performanceGuardSnapshot() const {
    QVariantMap map;
    map.insert(QStringLiteral("fpsTarget"), m_guard.fpsTarget);
    map.insert(QStringLiteral("reduceMotionAfter"), m_guard.reduceMotionAfterSeconds);
    map.insert(QStringLiteral("jankThresholdMs"), m_guard.jankThresholdMs);
    map.insert(QStringLiteral("maxOverlayCount"), m_guard.maxOverlayCount);
    map.insert(QStringLiteral("disableSecondaryWhenBelow"), m_guard.disableSecondaryWhenFpsBelow);
    return map;
}

QVariantMap Application::riskRefreshSnapshot() const
{
    QVariantMap map;
    map.insert(QStringLiteral("enabled"), m_riskRefreshEnabled);
    map.insert(QStringLiteral("intervalSeconds"), static_cast<double>(m_riskRefreshIntervalMs) / 1000.0);
    map.insert(QStringLiteral("active"), m_riskRefreshEnabled && m_riskRefreshTimer.isActive());
    map.insert(QStringLiteral("lastRequestAt"),
               m_lastRiskRefreshRequestUtc.isValid()
                   ? m_lastRiskRefreshRequestUtc.toString(Qt::ISODateWithMs)
                   : QString());
    map.insert(QStringLiteral("lastUpdateAt"),
               m_lastRiskUpdateUtc.isValid() ? m_lastRiskUpdateUtc.toString(Qt::ISODateWithMs) : QString());
    map.insert(QStringLiteral("nextRefreshDueAt"),
               m_nextRiskRefreshUtc.isValid() ? m_nextRiskRefreshUtc.toString(Qt::ISODateWithMs) : QString());
    double remainingSeconds = -1.0;
    if (m_nextRiskRefreshUtc.isValid()) {
        const qint64 remainingMs = QDateTime::currentDateTimeUtc().msecsTo(m_nextRiskRefreshUtc);
        remainingSeconds = remainingMs > 0 ? static_cast<double>(remainingMs) / 1000.0 : 0.0;
    }
    map.insert(QStringLiteral("nextRefreshInSeconds"), remainingSeconds);
    return map;
}

bool Application::updateInstrument(const QString& exchange,
                                   const QString& symbol,
                                   const QString& venueSymbol,
                                   const QString& quoteCurrency,
                                   const QString& baseCurrency,
                                   const QString& granularityIso8601) {
    TradingClient::InstrumentConfig config;
    config.exchange = exchange.trimmed();
    config.symbol = symbol.trimmed();
    config.venueSymbol = venueSymbol.trimmed();
    config.quoteCurrency = quoteCurrency.trimmed();
    config.baseCurrency = baseCurrency.trimmed();
    config.granularityIso8601 = granularityIso8601.trimmed();

    if (config.exchange.isEmpty() || config.symbol.isEmpty() || config.venueSymbol.isEmpty()
        || config.quoteCurrency.isEmpty() || config.baseCurrency.isEmpty()
        || config.granularityIso8601.isEmpty()) {
        qCWarning(lcAppMetrics) << "Odmowa aktualizacji instrumentu – wymagane pola są puste";
        return false;
    }

    const bool changed = config.exchange != m_instrument.exchange
        || config.symbol != m_instrument.symbol || config.venueSymbol != m_instrument.venueSymbol
        || config.quoteCurrency != m_instrument.quoteCurrency
        || config.baseCurrency != m_instrument.baseCurrency
        || config.granularityIso8601 != m_instrument.granularityIso8601;

    const bool wasStreaming = m_client.isStreaming();
    if (wasStreaming)
        m_client.stop();

    m_client.setInstrument(config);
    m_instrument = config;
    Q_EMIT instrumentChanged();

    if (wasStreaming)
        m_client.start();

    if (changed && !m_loadingUiSettings)
        scheduleUiSettingsPersist();

    return true;
}

bool Application::updatePerformanceGuard(int fpsTarget,
                                         double reduceMotionAfter,
                                         double jankThresholdMs,
                                         int maxOverlayCount,
                                         int disableSecondaryWhenBelow) {
    PerformanceGuard guard;
    guard.fpsTarget = qMax(1, fpsTarget);
    guard.reduceMotionAfterSeconds = qMax(0.0, reduceMotionAfter);
    guard.jankThresholdMs = qMax(0.0, jankThresholdMs);
    guard.maxOverlayCount = qMax(0, maxOverlayCount);
    guard.disableSecondaryWhenFpsBelow = qMax(0, disableSecondaryWhenBelow);

    const bool changed = guard.fpsTarget != m_guard.fpsTarget
        || !qFuzzyCompare(guard.reduceMotionAfterSeconds + 1.0, m_guard.reduceMotionAfterSeconds + 1.0)
        || !qFuzzyCompare(guard.jankThresholdMs + 1.0, m_guard.jankThresholdMs + 1.0)
        || guard.maxOverlayCount != m_guard.maxOverlayCount
        || guard.disableSecondaryWhenFpsBelow != m_guard.disableSecondaryWhenFpsBelow;

    m_client.setPerformanceGuard(guard);
    m_guard = guard;
    Q_EMIT performanceGuardChanged();

    if (m_frameMonitor)
        m_frameMonitor->setPerformanceGuard(m_guard);

    if (changed && !m_loadingUiSettings)
        scheduleUiSettingsPersist();

    return true;
}

bool Application::updateRiskRefresh(bool enabled, double intervalSeconds)
{
    const bool previousEnabled = m_riskRefreshEnabled;
    const int previousInterval = m_riskRefreshIntervalMs;

    double effectiveInterval = intervalSeconds;
    if (enabled) {
        if (!std::isfinite(intervalSeconds) || intervalSeconds <= 0.0) {
            qCWarning(lcAppMetrics)
                << "Odmowa aktualizacji harmonogramu ryzyka – oczekiwano dodatniego interwału (s), otrzymano"
                << intervalSeconds;
            return false;
        }
    } else if (!std::isfinite(intervalSeconds) || intervalSeconds <= 0.0) {
        effectiveInterval = static_cast<double>(m_riskRefreshIntervalMs) / 1000.0;
    }

    configureRiskRefresh(enabled, effectiveInterval);
    applyRiskRefreshTimerState();

    if (!m_loadingUiSettings
        && (previousEnabled != m_riskRefreshEnabled || previousInterval != m_riskRefreshIntervalMs)) {
        scheduleUiSettingsPersist();
    }

    return true;
}

bool Application::triggerRiskRefreshNow()
{
    const QDateTime requestTime = QDateTime::currentDateTimeUtc();
    m_lastRiskRefreshRequestUtc = requestTime;
    m_client.refreshRiskState();

    if (m_riskRefreshEnabled && m_started) {
        m_riskRefreshTimer.start(m_riskRefreshIntervalMs);
        m_nextRiskRefreshUtc = requestTime.addMSecs(m_riskRefreshIntervalMs);
    } else if (!m_riskRefreshEnabled) {
        m_nextRiskRefreshUtc = {};
    }

    Q_EMIT riskRefreshScheduleChanged();
    return true;
}

bool Application::updateRiskHistoryLimit(int maximumEntries)
{
    if (maximumEntries < 1) {
        qCWarning(lcAppMetrics)
            << "Odmowa aktualizacji limitu historii ryzyka – oczekiwano dodatniej liczby próbek, otrzymano"
            << maximumEntries;
        return false;
    }

    m_riskHistoryModel.setMaximumEntries(maximumEntries);

    return true;
}

void Application::clearRiskHistory()
{
    m_riskHistoryModel.clear();
}

bool Application::exportRiskHistoryToCsv(const QUrl& destination, int limit)
{
    if (limit == 0 || limit < -1) {
        qCWarning(lcAppMetrics)
            << "Nieprawidłowy limit eksportu historii ryzyka:" << limit;
        return false;
    }

    if (limit > 0)
        setRiskHistoryExportLimitValue(limit);

    setRiskHistoryExportLimitEnabled(limit > 0);

    if (!destination.isValid()) {
        qCWarning(lcAppMetrics)
            << "Nieprawidłowy adres docelowy eksportu historii ryzyka:" << destination;
        return false;
    }

    QString path;
    if (destination.isLocalFile() || destination.scheme().isEmpty()) {
        path = destination.isLocalFile() ? destination.toLocalFile() : destination.toString(QUrl::PreferLocalFile);
    } else {
        path = destination.toLocalFile();
    }

    if (path.trimmed().isEmpty()) {
        qCWarning(lcAppMetrics) << "Brak ścieżki docelowej do eksportu historii ryzyka";
        return false;
    }

    QFileInfo info(path);
    if (info.fileName().isEmpty()) {
        qCWarning(lcAppMetrics) << "Ścieżka eksportu historii ryzyka nie zawiera nazwy pliku:" << path;
        return false;
    }

    QDir directory = info.dir();
    if (!directory.exists() && !directory.mkpath(QStringLiteral("."))) {
        qCWarning(lcAppMetrics)
            << "Nie udało się utworzyć katalogu docelowego eksportu historii ryzyka:" << directory.absolutePath();
        return false;
    }

    const bool ok = m_riskHistoryModel.exportToCsv(info.absoluteFilePath(), limit);
    if (!ok) {
        qCWarning(lcAppMetrics) << "Eksport historii ryzyka nie powiódł się do pliku" << info.absoluteFilePath();
        return false;
    }

    setRiskHistoryExportLastDirectory(QUrl::fromLocalFile(info.dir().absolutePath()));

    return true;
}

bool Application::setRiskHistoryExportLimitEnabled(bool enabled)
{
    if (m_riskHistoryExportLimitEnabled == enabled)
        return true;

    m_riskHistoryExportLimitEnabled = enabled;
    Q_EMIT riskHistoryExportLimitEnabledChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryExportLimitValue(int limit)
{
    if (limit < 1) {
        qCWarning(lcAppMetrics)
            << "Odmowa ustawienia limitu eksportu historii ryzyka na wartość mniejszą niż 1:" << limit;
        return false;
    }

    if (m_riskHistoryExportLimitValue == limit)
        return true;

    m_riskHistoryExportLimitValue = limit;
    Q_EMIT riskHistoryExportLimitValueChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryExportLastDirectory(const QUrl& directory)
{
    if (!directory.isValid() && !directory.isEmpty())
        return false;

    QString path;
    if (directory.isLocalFile() || directory.scheme().isEmpty()) {
        path = directory.isLocalFile() ? directory.toLocalFile() : directory.toString(QUrl::PreferLocalFile);
    } else {
        return false;
    }

    path = path.trimmed();
    if (path.isEmpty())
        return false;

    const QString normalizedPath = QDir(path).absolutePath();
    const QUrl normalizedUrl = QUrl::fromLocalFile(normalizedPath);

    if (m_riskHistoryExportLastDirectory == normalizedUrl)
        return true;

    m_riskHistoryExportLastDirectory = normalizedUrl;
    m_riskHistoryAutoExportDirectoryWarned = false;
    Q_EMIT riskHistoryExportLastDirectoryChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryAutoExportEnabled(bool enabled)
{
    if (m_riskHistoryAutoExportEnabled == enabled)
        return true;

    m_riskHistoryAutoExportEnabled = enabled;
    Q_EMIT riskHistoryAutoExportEnabledChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryAutoExportIntervalMinutes(int minutes)
{
    if (minutes < 1) {
        qCWarning(lcAppMetrics)
            << "Odmowa ustawienia interwału auto-eksportu historii ryzyka na wartość mniejszą niż 1 minuta:" << minutes;
        return false;
    }

    if (m_riskHistoryAutoExportIntervalMinutes == minutes)
        return true;

    m_riskHistoryAutoExportIntervalMinutes = minutes;
    Q_EMIT riskHistoryAutoExportIntervalMinutesChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryAutoExportBasename(const QString& basename)
{
    const QString sanitized = sanitizeAutoExportBasename(basename);
    if (m_riskHistoryAutoExportBasename == sanitized)
        return true;

    m_riskHistoryAutoExportBasename = sanitized;
    Q_EMIT riskHistoryAutoExportBasenameChanged();
    scheduleUiSettingsPersist();
    return true;
}

bool Application::setRiskHistoryAutoExportUseLocalTime(bool useLocalTime)
{
    if (m_riskHistoryAutoExportUseLocalTime == useLocalTime)
        return true;

    m_riskHistoryAutoExportUseLocalTime = useLocalTime;
    Q_EMIT riskHistoryAutoExportUseLocalTimeChanged();
    scheduleUiSettingsPersist();
    return true;
}

void Application::saveUiSettingsImmediatelyForTesting()
{
    if (m_uiSettingsSaveTimer.isActive())
        m_uiSettingsSaveTimer.stop();
    persistUiSettings();
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

    if (parser.value("tls-pinned-sha256").trimmed().isEmpty()) {
        if (const auto pinnedEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_PINNED_SHA256")); pinnedEnv.has_value()) {
            m_tradingTlsConfig.pinnedServerFingerprint = pinnedEnv->trimmed();
        }
    }

    if (parser.value("grpc-target-name").trimmed().isEmpty()) {
        if (const auto targetEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_TARGET_NAME")))
            m_tradingTlsConfig.targetNameOverride = targetEnv->trimmed();
    }
}

void Application::applyTradingAuthEnvironmentOverrides(const QCommandLineParser& parser,
                                                       bool cliTokenProvided,
                                                       bool cliTokenFileProvided,
                                                       bool cliRoleProvided,
                                                       bool cliScopesProvided)
{
    Q_UNUSED(parser);

    if (!cliTokenProvided && !cliTokenFileProvided) {
        const auto envToken = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_AUTH_TOKEN"));
        const auto envTokenFile = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_AUTH_TOKEN_FILE"));
        const bool envTokenNonEmpty = envToken.has_value() && !envToken->trimmed().isEmpty();
        const bool envTokenFileNonEmpty = envTokenFile.has_value() && !envTokenFile->trimmed().isEmpty();

        if (envTokenNonEmpty && envTokenFileNonEmpty) {
            qCWarning(lcAppMetrics)
                << "Zmiennie BOT_CORE_UI_GRPC_AUTH_TOKEN oraz BOT_CORE_UI_GRPC_AUTH_TOKEN_FILE są ustawione równocześnie."
                << "Użyję wartości tokenu przekazanej w zmiennej BOT_CORE_UI_GRPC_AUTH_TOKEN.";
        }

        bool applied = false;
        if (envToken.has_value()) {
            const QString trimmed = envToken->trimmed();
            if (!trimmed.isEmpty()) {
                m_tradingAuthToken = trimmed;
                applied = true;
            }
        }

        if (!applied && envTokenFileNonEmpty) {
            const QString tokenFromFile = readTokenFile(envTokenFile->trimmed(), QStringLiteral("TradingService"));
            if (!tokenFromFile.isEmpty()) {
                m_tradingAuthToken = tokenFromFile;
                applied = true;
            }
        }

        if (!applied && ((envToken.has_value() && envToken->trimmed().isEmpty())
                         || (envTokenFile.has_value() && envTokenFile->trimmed().isEmpty()))) {
            m_tradingAuthToken.clear();
        }
    }

    if (!cliRoleProvided) {
        if (const auto roleEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_RBAC_ROLE")); roleEnv.has_value()) {
            m_tradingRbacRole = roleEnv->trimmed();
        }
    }

    if (!cliScopesProvided) {
        if (const auto scopesEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_GRPC_RBAC_SCOPES")); scopesEnv.has_value()) {
            const QString trimmed = scopesEnv->trimmed();
            if (trimmed.isEmpty()) {
                m_tradingRbacScopes.clear();
            } else {
                m_tradingRbacScopes = splitScopesList(trimmed);
            }
        }
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

void Application::startRiskRefreshTimerForTesting()
{
    m_started = true;
    applyRiskRefreshTimerState();
}

void Application::setLastRiskHistoryAutoExportForTesting(const QDateTime& timestamp)
{
    if (timestamp.isValid())
        m_lastRiskHistoryAutoExportUtc = timestamp.toUTC();
    else
        m_lastRiskHistoryAutoExportUtc = {};
    Q_EMIT riskHistoryLastAutoExportAtChanged();
}

void Application::setRiskHistoryAutoExportLastPathForTesting(const QUrl& url)
{
    m_lastRiskHistoryAutoExportPath = url;
    Q_EMIT riskHistoryLastAutoExportPathChanged();
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

    if (m_metricsRbacRole.trimmed().isEmpty()) {
        if (const auto roleEnv = envValue(QByteArrayLiteral("BOT_CORE_UI_METRICS_RBAC_ROLE")); roleEnv.has_value()) {
            m_metricsRbacRole = roleEnv->trimmed();
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
        if (!shouldEnable) {
            updateTelemetryPendingRetryCount(0);
            return;
        }
        auto reporter = std::make_unique<UiTelemetryReporter>(this);
        m_telemetry = std::move(reporter);
    }
    if (!m_telemetry)
        return;

    if (auto* uiReporter = dynamic_cast<UiTelemetryReporter*>(m_telemetry.get())) {
        connect(uiReporter, &UiTelemetryReporter::pendingRetryCountChanged,
                this, &Application::handleTelemetryPendingRetryCountChanged,
                Qt::UniqueConnection);
        updateTelemetryPendingRetryCount(uiReporter->pendingRetryCount());
    } else {
        updateTelemetryPendingRetryCount(0);
    }

    m_telemetry->setWindowCount(m_windowCount);
    m_telemetry->setNotesTag(m_metricsTag);
    m_telemetry->setEndpoint(m_metricsEndpoint);
    m_telemetry->setTlsConfig(m_tlsConfig);
    m_telemetry->setAuthToken(m_metricsAuthToken);
    m_telemetry->setRbacRole(m_metricsRbacRole);
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

void Application::handleTelemetryPendingRetryCountChanged(int pending) {
    updateTelemetryPendingRetryCount(pending);
}

void Application::updateTelemetryPendingRetryCount(int pending) {
    if (m_pendingRetryCount == pending)
        return;
    m_pendingRetryCount = pending;
    Q_EMIT telemetryPendingRetryCountChanged(pending);
}
