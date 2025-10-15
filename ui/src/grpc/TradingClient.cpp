#include "TradingClient.hpp"

#include <QDateTime>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QIODevice>
#include <QMetaObject>
#include <QtGlobal>
#include <QSslCertificate>

#include <google/protobuf/timestamp.pb.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/channel_arguments.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/security/credentials.h>

#include <optional>
#include <string>

#include "trading.grpc.pb.h"

using botcore::trading::v1::GetOhlcvHistoryRequest;
using botcore::trading::v1::GetOhlcvHistoryResponse;
using botcore::trading::v1::Instrument;
using botcore::trading::v1::MarketDataService;
using botcore::trading::v1::OhlcvCandle;
using botcore::trading::v1::StreamOhlcvRequest;
using botcore::trading::v1::StreamOhlcvUpdate;
using botcore::trading::v1::RiskService;
using botcore::trading::v1::RiskState;

namespace {

qint64 timestampToMs(const google::protobuf::Timestamp& ts) {
    return static_cast<qint64>(ts.seconds()) * 1000 + ts.nanos() / 1000000;
}

QString expandUserPath(const QString& path) {
    if (path == QStringLiteral("~")) {
        return QDir::homePath();
    }
    if (path.startsWith(QStringLiteral("~/"))) {
        return QDir::homePath() + path.mid(1);
    }
    return path;
}

std::optional<QByteArray> readFileUtf8(const QString& rawPath) {
    const QString path = expandUserPath(rawPath);
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
    m_tlsConfig = config;
    // zmiana TLS wymaga odtworzenia kanału/stubów
    m_channel.reset();
    m_marketDataStub.reset();
    m_riskStub.reset();
}

void TradingClient::start() {
    if (m_running.exchange(true)) {
        return;
    }
    if (m_streamThread.joinable()) {
        m_streamThread.join();
    }
    ensureStub();

    Q_EMIT streamingChanged();
    Q_EMIT connectionStateChanged(tr("connecting"));

    GetOhlcvHistoryRequest historyReq;
    *historyReq.mutable_instrument() = makeInstrument(m_instrumentConfig);
    historyReq.mutable_granularity()->set_iso8601_duration(m_instrumentConfig.granularityIso8601.toStdString());
    historyReq.set_limit(m_historyLimit);

    GetOhlcvHistoryResponse historyResp;
    grpc::ClientContext historyContext;
    const grpc::Status historyStatus = m_marketDataStub->GetOhlcvHistory(&historyContext, historyReq, &historyResp);
    if (historyStatus.ok()) {
        Q_EMIT historyReceived(convertHistory(historyResp.candles()));
    } else {
        Q_EMIT connectionStateChanged(QStringLiteral("history error: %1")
                                          .arg(QString::fromStdString(historyStatus.error_message())));
    }

    refreshRiskState();

    {
        std::lock_guard<std::mutex> lock(m_contextMutex);
        m_activeContext = std::make_shared<grpc::ClientContext>();
    }

    m_streamThread = std::thread([this]() { streamLoop(); });
}

void TradingClient::stop() {
    const bool wasRunning = m_running.exchange(false);
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
    if (!m_channel) {
        std::shared_ptr<grpc::ChannelCredentials> credentials;
        grpc::ChannelArguments args;

        if (m_tlsConfig.enabled) {
            grpc::SslCredentialsOptions options;

            if (const auto rootPem = readFileUtf8(m_tlsConfig.rootCertificatePath)) {
                options.pem_root_certs = std::string(rootPem->constData(),
                                                     static_cast<std::size_t>(rootPem->size()));
            }

            const auto clientCert = readFileUtf8(m_tlsConfig.clientCertificatePath);
            const auto clientKey  = readFileUtf8(m_tlsConfig.clientKeyPath);
            if (clientCert && clientKey) {
                grpc::SslCredentialsOptions::PemKeyCertPair pair;
                pair.private_key = std::string(clientKey->constData(),  static_cast<std::size_t>(clientKey->size()));
                pair.cert_chain  = std::string(clientCert->constData(), static_cast<std::size_t>(clientCert->size()));
                options.pem_key_cert_pairs.push_back(std::move(pair));
            }

            credentials = grpc::SslCredentials(options);

            // Uwaga: w TlsConfig używamy targetNameOverride (zgodne z Application.cpp)
            if (!m_tlsConfig.targetNameOverride.trimmed().isEmpty()) {
                args.SetString(GRPC_SSL_TARGET_NAME_OVERRIDE_ARG,
                               m_tlsConfig.targetNameOverride.toStdString());
            }

            m_channel = grpc::CreateCustomChannel(m_endpoint.toStdString(), credentials, args);
        } else {
            credentials = grpc::InsecureChannelCredentials();
            m_channel = grpc::CreateCustomChannel(m_endpoint.toStdString(), credentials, args);
        }
    }

    m_marketDataStub = MarketDataService::NewStub(m_channel);
    m_riskStub = RiskService::NewStub(m_channel);
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

void TradingClient::streamLoop() {
    StreamOhlcvRequest request;
    *request.mutable_instrument() = makeInstrument(m_instrumentConfig);
    request.mutable_granularity()->set_iso8601_duration(m_instrumentConfig.granularityIso8601.toStdString());
    request.set_deliver_snapshots(true);

    std::shared_ptr<grpc::ClientContext> context;
    {
        std::lock_guard<std::mutex> lock(m_contextMutex);
        context = m_activeContext;
    }

    auto reader = m_marketDataStub->StreamOhlcv(context.get(), request);
    StreamOhlcvUpdate update;

    while (m_running.load() && reader->Read(&update)) {
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

    const grpc::Status finishStatus = reader->Finish();
    if (!finishStatus.ok() && m_running.load()) {
        QMetaObject::invokeMethod(
            this,
            [this, finishStatus]() {
                Q_EMIT connectionStateChanged(QStringLiteral("stream error: %1")
                                                  .arg(QString::fromStdString(finishStatus.error_message())));
            },
            Qt::QueuedConnection);
    } else {
        QMetaObject::invokeMethod(
            this,
            [this]() { Q_EMIT connectionStateChanged(tr("stream ended")); },
            Qt::QueuedConnection);
    }

    {
        std::lock_guard<std::mutex> lock(m_contextMutex);
        m_activeContext.reset();
    }
    m_running.store(false);
    QMetaObject::invokeMethod(this, [this]() { Q_EMIT streamingChanged(); }, Qt::QueuedConnection);
}

void TradingClient::refreshRiskState() {
    if (!m_riskStub) {
        return;
    }
    grpc::ClientContext riskContext;
    botcore::trading::v1::RiskStateRequest request;
    RiskState response;
    const grpc::Status status = m_riskStub->GetRiskState(&riskContext, request, &response);
    if (status.ok()) {
        const auto snapshot = convertRiskState(response);
        QMetaObject::invokeMethod(
            this,
            [this, snapshot]() { Q_EMIT riskStateReceived(snapshot); },
            Qt::QueuedConnection);
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

TradingClient::PreLiveChecklistResult TradingClient::runPreLiveChecklist() const {
    PreLiveChecklistResult result;

    if (m_endpoint.trimmed().isEmpty()) {
        result.errors.append(tr("Endpoint gRPC nie może być pusty."));
    }

    if (m_tlsConfig.enabled) {
        const QString rootPath = expandUserPath(m_tlsConfig.rootCertificatePath);
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

        const QString clientCert = expandUserPath(m_tlsConfig.clientCertificatePath);
        const QString clientKey = expandUserPath(m_tlsConfig.clientKeyPath);
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
