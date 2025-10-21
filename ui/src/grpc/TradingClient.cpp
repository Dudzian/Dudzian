#include "TradingClient.hpp"

#include <QDateTime>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QIODevice>
#include <QMetaObject>
#include <QLoggingCategory>
#include <QSet>
#include <QtGlobal>
#include <QSslCertificate>

#include <google/protobuf/timestamp.pb.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/channel_arguments.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/security/credentials.h>

#include <algorithm>
#include <chrono>
#include <optional>
#include <string>
#include <thread>

#include "trading.grpc.pb.h"
#include "utils/PathUtils.hpp"

Q_LOGGING_CATEGORY(lcTradingClient, "bot.shell.trading.grpc")

Q_LOGGING_CATEGORY(lcTradingClient, "bot.shell.trading.grpc")

using botcore::trading::v1::GetOhlcvHistoryRequest;
using botcore::trading::v1::GetOhlcvHistoryResponse;
using botcore::trading::v1::Instrument;
using botcore::trading::v1::MarketDataService;
using botcore::trading::v1::OhlcvCandle;
using botcore::trading::v1::StreamOhlcvRequest;
using botcore::trading::v1::StreamOhlcvUpdate;
using botcore::trading::v1::RiskService;
using botcore::trading::v1::RiskState;
using botcore::trading::v1::RiskStateRequest;

namespace {

qint64 timestampToMs(const google::protobuf::Timestamp& ts) {
    return static_cast<qint64>(ts.seconds()) * 1000 + ts.nanos() / 1000000;
}

std::optional<QByteArray> readFileUtf8(const QString& rawPath) {
    const QString path = bot::shell::utils::expandPath(rawPath);
    if (path.trimmed().isEmpty()) {
        return std::nullopt;
    }
    QFile file(path);
    if (!file.exists()) {
        return std::nullopt;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return std::nullopt;
    }
    return file.readAll();
}

QString sha256Fingerprint(const QByteArray& pemData) {
    const QList<QSslCertificate> certs = QSslCertificate::fromData(pemData, QSsl::Pem);
    if (certs.isEmpty()) {
        return {};
    }
    const QByteArray digest = certs.first().digest(QCryptographicHash::Sha256);
    return QString::fromLatin1(digest.toHex()).toLower();
}

QString normalizeFingerprint(QString value) {
    QString normalized = value.trimmed().toLower();
    normalized.remove(QLatin1Char(':'));
    return normalized;
}

Instrument makeInstrument(const TradingClient::InstrumentConfig& config) {
    Instrument instrument;
    instrument.set_exchange(config.exchange.toStdString());
    instrument.set_symbol(config.symbol.toStdString());
    instrument.set_venue_symbol(config.venueSymbol.toStdString());
    instrument.set_quote_currency(config.quoteCurrency.toStdString());
    instrument.set_base_currency(config.baseCurrency.toStdString());
    return instrument;
}

} // namespace

TradingClient::TradingClient(QObject* parent)
    : QObject(parent) {
    qRegisterMetaType<QList<OhlcvPoint>>("QList<OhlcvPoint>");
    qRegisterMetaType<PerformanceGuard>("PerformanceGuard");
    qRegisterMetaType<RiskSnapshotData>("RiskSnapshotData");
}

TradingClient::~TradingClient() {
    stop();
}

void TradingClient::setEndpoint(const QString& endpoint) {
    if (endpoint == m_endpoint) {
        return;
    }
    m_endpoint = endpoint;
    m_channel.reset();
    m_marketDataStub.reset();
    m_riskStub.reset();
}

void TradingClient::setInstrument(const InstrumentConfig& config) {
    m_instrumentConfig = config;
}

void TradingClient::setHistoryLimit(int limit) {
    if (limit > 0) {
        m_historyLimit = limit;
    }
}

void TradingClient::setPerformanceGuard(const PerformanceGuard& guard) {
    m_guard = guard;
    Q_EMIT performanceGuardUpdated(m_guard);
}

void TradingClient::setTlsConfig(const TlsConfig& config) {
    TlsConfig sanitized = config;
    sanitized.pinnedServerFingerprint = normalizeFingerprint(sanitized.pinnedServerFingerprint);
    m_tlsConfig = sanitized;
    // zmiana TLS wymaga odtworzenia kanału/stubów
    m_channel.reset();
    m_marketDataStub.reset();
    m_riskStub.reset();
}

void TradingClient::setAuthToken(const QString& token)
{
    const QString sanitized = token.trimmed();
    {
        std::lock_guard<std::mutex> lock(m_authMutex);
        if (m_authToken == sanitized)
            return;
        m_authToken = sanitized;
    }
    triggerStreamRestart();
}

void TradingClient::setRbacRole(const QString& role)
{
    const QString sanitized = role.trimmed();
    {
        std::lock_guard<std::mutex> lock(m_authMutex);
        if (m_rbacRole == sanitized)
            return;
        m_rbacRole = sanitized;
    }
    triggerStreamRestart();
}

void TradingClient::setRbacScopes(const QStringList& scopes)
{
    QStringList sanitized;
    QSet<QString> seen;
    for (const QString& scope : scopes) {
        const QString trimmed = scope.trimmed();
        if (trimmed.isEmpty())
            continue;
        if (seen.contains(trimmed))
            continue;
        seen.insert(trimmed);
        sanitized.append(trimmed);
    }

    {
        std::lock_guard<std::mutex> lock(m_authMutex);
        if (m_rbacScopes == sanitized)
            return;
        m_rbacScopes = sanitized;
    }
    triggerStreamRestart();
}

QVector<QPair<QByteArray, QByteArray>> TradingClient::authMetadataForTesting() const
{
    QVector<QPair<QByteArray, QByteArray>> result;
    const auto metadata = buildAuthMetadata();
    result.reserve(static_cast<int>(metadata.size()));
    for (const auto& entry : metadata) {
        result.append({QByteArray::fromStdString(entry.first), QByteArray::fromStdString(entry.second)});
    }
    return result;
}

void TradingClient::start() {
    if (m_running.exchange(true)) {
        return;
    }
    if (m_streamThread.joinable()) {
        m_streamThread.join();
    }
    m_restartRequested.store(false);
    ensureStub();

    if (!m_marketDataStub) {
        qCWarning(lcTradingClient) << "Brak poprawnie zainicjalizowanego stubu MarketDataService dla endpointu" << m_endpoint;
        m_running.store(false);
        Q_EMIT streamingChanged();
        Q_EMIT connectionStateChanged(tr("unavailable"));
        return;
    }

    Q_EMIT streamingChanged();
    Q_EMIT connectionStateChanged(tr("connecting"));

    GetOhlcvHistoryRequest historyReq;
    *historyReq.mutable_instrument() = makeInstrument(m_instrumentConfig);
    historyReq.mutable_granularity()->set_iso8601_duration(m_instrumentConfig.granularityIso8601.toStdString());
    historyReq.set_limit(m_historyLimit);

    GetOhlcvHistoryResponse historyResp;
    auto historyContext = createContext();
    const grpc::Status historyStatus = m_marketDataStub->GetOhlcvHistory(historyContext.get(), historyReq, &historyResp);
    if (historyStatus.ok()) {
        Q_EMIT historyReceived(convertHistory(historyResp.candles()));
    } else {
        Q_EMIT connectionStateChanged(QStringLiteral("history error: %1")
                                          .arg(QString::fromStdString(historyStatus.error_message())));
    }

    refreshRiskState();

    {
        std::lock_guard<std::mutex> lock(m_contextMutex);
        m_activeContext = createContext();
    }

    m_streamThread = std::thread([this]() { streamLoop(); });
}

void TradingClient::stop() {
    const bool wasRunning = m_running.exchange(false);
    m_restartRequested.store(false);
    {
        std::lock_guard<std::mutex> lock(m_contextMutex);
        if (m_activeContext) {
            m_activeContext->TryCancel();
        }
    }
    if (m_streamThread.joinable()) {
        m_streamThread.join();
    }
    if (wasRunning) {
        Q_EMIT streamingChanged();
        Q_EMIT connectionStateChanged(tr("stopped"));
    }
}

void TradingClient::ensureStub() {
    if (m_endpoint.trimmed().isEmpty()) {
        qCWarning(lcTradingClient) << "Endpoint gRPC nie został ustawiony – pomijam inicjalizację kanału.";
        return;
    }

    if (!m_channel) {
        std::shared_ptr<grpc::ChannelCredentials> credentials;
        grpc::ChannelArguments args;
        QByteArray rootPem;
        bool fingerprintValid = true;

        if (m_tlsConfig.enabled) {
            grpc::SslCredentialsOptions options;

            if (!m_tlsConfig.rootCertificatePath.trimmed().isEmpty()) {
                if (const auto rootData = readFileUtf8(m_tlsConfig.rootCertificatePath)) {
                    rootPem = *rootData;
                    options.pem_root_certs = std::string(rootPem.constData(),
                                                         static_cast<std::size_t>(rootPem.size()));
                } else {
                    qCWarning(lcTradingClient) << "Nie udało się odczytać pliku root CA" << m_tlsConfig.rootCertificatePath;
                }
            } else {
                qCWarning(lcTradingClient) << "TLS aktywny bez wskazanego pliku root CA.";
            }

            const auto clientCert = readFileUtf8(m_tlsConfig.clientCertificatePath);
            const auto clientKey  = readFileUtf8(m_tlsConfig.clientKeyPath);
            if (clientCert && clientKey) {
                grpc::SslCredentialsOptions::PemKeyCertPair pair;
                pair.private_key = std::string(clientKey->constData(), static_cast<std::size_t>(clientKey->size()));
                pair.cert_chain  = std::string(clientCert->constData(), static_cast<std::size_t>(clientCert->size()));
                options.pem_key_cert_pairs.push_back(std::move(pair));
            } else if (m_tlsConfig.requireClientAuth) {
                qCWarning(lcTradingClient) << "mTLS wymaga zarówno certyfikatu, jak i klucza klienta.";
                fingerprintValid = false;
            }

            if (!m_tlsConfig.pinnedServerFingerprint.isEmpty()) {
                if (rootPem.isEmpty()) {
                    qCWarning(lcTradingClient) << "Nie mogę zweryfikować fingerprintu TLS – brak danych root CA.";
                    fingerprintValid = false;
                } else {
                    const QString actual = sha256Fingerprint(rootPem);
                    if (actual.isEmpty()) {
                        qCWarning(lcTradingClient) << "Nie udało się obliczyć fingerprintu SHA-256 certyfikatu root.";
                        fingerprintValid = false;
                    } else if (actual != m_tlsConfig.pinnedServerFingerprint) {
                        qCWarning(lcTradingClient)
                            << "Fingerprint TLS nie pasuje do konfiguracji (oczekiwano"
                            << m_tlsConfig.pinnedServerFingerprint << "otrzymano" << actual << ')';
                        fingerprintValid = false;
                    }
                }
            }

            if (!fingerprintValid) {
                return;
            }

            credentials = grpc::SslCredentials(options);

            if (!m_tlsConfig.targetNameOverride.trimmed().isEmpty()) {
                args.SetString(GRPC_SSL_TARGET_NAME_OVERRIDE_ARG,
                               m_tlsConfig.targetNameOverride.toStdString());
            }

            m_channel = grpc::CreateCustomChannel(m_endpoint.toStdString(), credentials, args);
        } else {
            if (!m_tlsConfig.pinnedServerFingerprint.isEmpty()) {
                qCWarning(lcTradingClient)
                    << "Podano fingerprint TLS, ale połączenie TLS jest wyłączone – pinning zostanie zignorowany.";
            }
            credentials = grpc::InsecureChannelCredentials();
            m_channel = grpc::CreateCustomChannel(m_endpoint.toStdString(), credentials, args);
        }
    }

    if (m_channel && !m_marketDataStub) {
        m_marketDataStub = MarketDataService::NewStub(m_channel);
    }
    if (m_channel && !m_riskStub) {
        m_riskStub = RiskService::NewStub(m_channel);
    }
}

QList<OhlcvPoint> TradingClient::convertHistory(const google::protobuf::RepeatedPtrField<OhlcvCandle>& candles) const {
    QList<OhlcvPoint> result;
    result.reserve(static_cast<int>(candles.size()));
    for (const auto& candle : candles) {
        result.append(convertCandle(candle));
    }
    return result;
}

OhlcvPoint TradingClient::convertCandle(const OhlcvCandle& candle) const {
    OhlcvPoint point;
    point.timestampMs = timestampToMs(candle.open_time());
    point.open = candle.open();
    point.high = candle.high();
    point.low = candle.low();
    point.close = candle.close();
    point.volume = candle.volume();
    point.closed = candle.closed();
    point.sequence = candle.sequence();
    return point;
}

void TradingClient::streamLoop()
{
    int attempt = 0;

    while (m_running.load()) {
        ensureStub();
        if (!m_marketDataStub) {
            QMetaObject::invokeMethod(
                this,
                [this]() { Q_EMIT connectionStateChanged(tr("stream unavailable")); },
                Qt::QueuedConnection);
            break;
        }

        std::shared_ptr<grpc::ClientContext> context;
        {
            std::lock_guard<std::mutex> lock(m_contextMutex);
            if (!m_activeContext) {
                m_activeContext = createContext();
            }
            context = m_activeContext;
        }

        StreamOhlcvRequest request;
        *request.mutable_instrument() = makeInstrument(m_instrumentConfig);
        request.mutable_granularity()->set_iso8601_duration(m_instrumentConfig.granularityIso8601.toStdString());
        request.set_deliver_snapshots(true);

        if (attempt > 0) {
            QMetaObject::invokeMethod(
                this,
                [this, attempt]() { Q_EMIT connectionStateChanged(tr("reconnecting (%1)").arg(attempt)); },
                Qt::QueuedConnection);
        } else {
            QMetaObject::invokeMethod(
                this,
                [this]() { Q_EMIT connectionStateChanged(tr("streaming")); },
                Qt::QueuedConnection);
        }

        auto reader = m_marketDataStub->StreamOhlcv(context.get(), request);
        StreamOhlcvUpdate update;
        bool receivedAny = false;

        while (m_running.load() && reader->Read(&update)) {
            receivedAny = true;
            attempt = 0;
            if (update.has_snapshot()) {
                const auto history = convertHistory(update.snapshot().candles());
                QMetaObject::invokeMethod(
                    this,
                    [this, history]() { Q_EMIT historyReceived(history); },
                    Qt::QueuedConnection);
            }
            if (update.has_increment()) {
                const auto point = convertCandle(update.increment().candle());
                QMetaObject::invokeMethod(
                    this,
                    [this, point]() { Q_EMIT candleReceived(point); },
                    Qt::QueuedConnection);
            }
        }

        const grpc::Status status = reader->Finish();

        {
            std::lock_guard<std::mutex> lock(m_contextMutex);
            if (m_activeContext == context) {
                m_activeContext.reset();
            }
        }

        if (!m_running.load()) {
            break;
        }

        const bool restartRequested = m_restartRequested.exchange(false);

        if (status.ok()) {
            attempt = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            continue;
        }

        if (status.error_code() == grpc::StatusCode::CANCELLED && restartRequested) {
            attempt = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (status.error_code() == grpc::StatusCode::CANCELLED && !receivedAny) {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            continue;
        }

        QMetaObject::invokeMethod(
            this,
            [this, status]() {
                Q_EMIT connectionStateChanged(QStringLiteral("stream error: %1")
                                                  .arg(QString::fromStdString(status.error_message())));
            },
            Qt::QueuedConnection);

        ++attempt;
        const int backoffMs = std::min(5000, 500 * std::max(1, attempt));
        std::this_thread::sleep_for(std::chrono::milliseconds(backoffMs));
    }

    m_running.store(false);
    QMetaObject::invokeMethod(this, [this]() { Q_EMIT streamingChanged(); }, Qt::QueuedConnection);
}

void TradingClient::refreshRiskState() {
    ensureStub();
    if (!m_riskStub) {
        qCWarning(lcTradingClient) << "Brak stubu RiskService – pomijam odczyt stanu ryzyka.";
        return;
    }
    auto riskContext = createContext();
    RiskStateRequest request;
    RiskState response;
    const grpc::Status status = m_riskStub->GetRiskState(riskContext.get(), request, &response);
    if (status.ok()) {
        const auto snapshot = convertRiskState(response);
        QMetaObject::invokeMethod(
            this,
            [this, snapshot]() { Q_EMIT riskStateReceived(snapshot); },
            Qt::QueuedConnection);
    } else {
        qCWarning(lcTradingClient)
            << "GetRiskState nie powiodło się:" << QString::fromStdString(status.error_message());
    }
}

RiskSnapshotData TradingClient::convertRiskState(const RiskState& state) const {
    RiskSnapshotData snapshot;
    snapshot.hasData = true;
    snapshot.profileEnum = state.profile();
    switch (state.profile()) {
    case botcore::trading::v1::RiskProfile::RISK_PROFILE_CONSERVATIVE:
        snapshot.profileLabel = QStringLiteral("Konserwatywny");
        break;
    case botcore::trading::v1::RiskProfile::RISK_PROFILE_BALANCED:
        snapshot.profileLabel = QStringLiteral("Zbalansowany");
        break;
    case botcore::trading::v1::RiskProfile::RISK_PROFILE_AGGRESSIVE:
        snapshot.profileLabel = QStringLiteral("Agresywny");
        break;
    case botcore::trading::v1::RiskProfile::RISK_PROFILE_MANUAL:
        snapshot.profileLabel = QStringLiteral("Manualny");
        break;
    default:
        snapshot.profileLabel = QStringLiteral("Nieokreślony");
        break;
    }
    snapshot.portfolioValue = state.portfolio_value();
    snapshot.currentDrawdown = state.current_drawdown();
    snapshot.maxDailyLoss = state.max_daily_loss();
    snapshot.usedLeverage = state.used_leverage();
    if (state.has_generated_at()) {
        const auto ts = state.generated_at();
        snapshot.generatedAt = QDateTime::fromSecsSinceEpoch(ts.seconds(), Qt::UTC);
        snapshot.generatedAt = snapshot.generatedAt.addMSecs(ts.nanos() / 1000000);
    }
    snapshot.exposures.reserve(static_cast<int>(state.limits_size()));
    for (const auto& limit : state.limits()) {
        RiskExposureData exposure;
        exposure.code = QString::fromStdString(limit.code());
        exposure.maxValue = limit.max_value();
        exposure.currentValue = limit.current_value();
        exposure.thresholdValue = limit.threshold_value();
        snapshot.exposures.append(exposure);
    }
    return snapshot;
}

std::shared_ptr<grpc::ClientContext> TradingClient::createContext() const
{
    auto context = std::make_shared<grpc::ClientContext>();
    applyAuthMetadata(*context);
    return context;
}

std::vector<std::pair<std::string, std::string>> TradingClient::buildAuthMetadata() const
{
    QString token;
    QString role;
    QStringList scopes;
    {
        std::lock_guard<std::mutex> lock(m_authMutex);
        token = m_authToken;
        role = m_rbacRole;
        scopes = m_rbacScopes;
    }

    std::vector<std::pair<std::string, std::string>> metadata;
    metadata.reserve(2 + scopes.size());

    if (!token.isEmpty()) {
        metadata.emplace_back("authorization", std::string("Bearer ") + token.toStdString());
    }
    if (!role.isEmpty()) {
        metadata.emplace_back("x-bot-role", role.toStdString());
    }
    for (const QString& scope : scopes) {
        metadata.emplace_back("x-bot-scope", scope.toStdString());
    }
    return metadata;
}

void TradingClient::applyAuthMetadata(grpc::ClientContext& context) const
{
    const auto metadata = buildAuthMetadata();
    for (const auto& entry : metadata) {
        context.AddMetadata(entry.first, entry.second);
    }
    context.AddMetadata("x-bot-channel", "desktop-ui");
}

void TradingClient::triggerStreamRestart()
{
    if (!m_running.load()) {
        return;
    }
    m_restartRequested.store(true);
    std::lock_guard<std::mutex> lock(m_contextMutex);
    if (m_activeContext) {
        m_activeContext->TryCancel();
    }
}

TradingClient::PreLiveChecklistResult TradingClient::runPreLiveChecklist() const {
    PreLiveChecklistResult result;

    if (m_endpoint.trimmed().isEmpty()) {
        result.errors.append(tr("Endpoint gRPC nie może być pusty."));
    }

    if (m_tlsConfig.enabled) {
        const QString rootPath = bot::shell::utils::expandPath(m_tlsConfig.rootCertificatePath);
        if (rootPath.trimmed().isEmpty()) {
            result.errors.append(tr("Włączone TLS wymaga wskazania pliku root CA."));
        } else if (!QFile::exists(rootPath)) {
            result.errors.append(tr("Plik root CA nie istnieje: %1").arg(rootPath));
        } else {
            const auto rootPem = readFileUtf8(rootPath);
            if (!rootPem) {
                result.warnings.append(tr("Nie udało się odczytać pliku root CA."));
            } else if (!m_tlsConfig.pinnedServerFingerprint.isEmpty()) {
                const QString actual = sha256Fingerprint(*rootPem);
                const QString expected = normalizeFingerprint(m_tlsConfig.pinnedServerFingerprint);
                if (actual.isEmpty()) {
                    result.warnings.append(tr("Nie udało się obliczyć odcisku SHA-256 certyfikatu root CA."));
                } else if (actual != expected) {
                    result.errors.append(tr("Fingerprint root CA nie pasuje do konfiguracji (oczekiwano %1, otrzymano %2).").arg(expected, actual));
                }
            }
        }

        const QString clientCert = bot::shell::utils::expandPath(m_tlsConfig.clientCertificatePath);
        const QString clientKey = bot::shell::utils::expandPath(m_tlsConfig.clientKeyPath);
        const bool hasClientCert = !clientCert.trimmed().isEmpty();
        const bool hasClientKey = !clientKey.trimmed().isEmpty();

        if (m_tlsConfig.requireClientAuth && (!hasClientCert || !hasClientKey)) {
            result.errors.append(tr("mTLS wymaga dostarczenia certyfikatu oraz klucza klienta."));
        } else {
            if (hasClientCert && !QFile::exists(clientCert)) {
                result.errors.append(tr("Certyfikat klienta nie istnieje: %1").arg(clientCert));
            }
            if (hasClientKey && !QFile::exists(clientKey)) {
                result.errors.append(tr("Klucz klienta nie istnieje: %1").arg(clientKey));
            }
        }
    }

    result.ok = result.errors.isEmpty();
    return result;
}
