#include "TradingClient.hpp"

#include <QDateTime>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QIODevice>
#include <QTextStream>
#include <QMetaObject>
#include <QLoggingCategory>
#include <QSet>
#include <QVector>
#include <QtGlobal>
#include <QSslCertificate>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QJsonValue>

#include <google/protobuf/timestamp.pb.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/channel_arguments.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/security/credentials.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>

#include "models/MarketRegimeClassifierBridge.hpp"

#include "trading.grpc.pb.h"
#include "utils/PathUtils.hpp"

Q_LOGGING_CATEGORY(lcTradingClient, "bot.shell.trading.grpc")

namespace {

using botcore::trading::v1::GetOhlcvHistoryRequest;
using botcore::trading::v1::GetOhlcvHistoryResponse;
using botcore::trading::v1::Instrument;
using botcore::trading::v1::ListTradableInstrumentsRequest;
using botcore::trading::v1::ListTradableInstrumentsResponse;
using botcore::trading::v1::OhlcvCandle;
using botcore::trading::v1::RiskState;
using botcore::trading::v1::RiskStateRequest;
using botcore::trading::v1::StreamOhlcvRequest;
using botcore::trading::v1::StreamOhlcvUpdate;

class MarketDataStreamReader {
public:
    virtual ~MarketDataStreamReader() = default;
    virtual bool Read(StreamOhlcvUpdate* update) = 0;
    virtual grpc::Status Finish() = 0;
};

class TradingClient::MarketDataStubInterface {
public:
    virtual ~MarketDataStubInterface() = default;
    virtual grpc::Status GetOhlcvHistory(grpc::ClientContext* context,
                                         const GetOhlcvHistoryRequest& request,
                                         GetOhlcvHistoryResponse* response) = 0;
    virtual std::unique_ptr<MarketDataStreamReader> StreamOhlcv(
        grpc::ClientContext* context,
        const StreamOhlcvRequest& request) = 0;
    virtual grpc::Status ListTradableInstruments(
        grpc::ClientContext* context,
        const ListTradableInstrumentsRequest& request,
        ListTradableInstrumentsResponse* response) = 0;
};

class TradingClient::RiskServiceStubInterface {
public:
    virtual ~RiskServiceStubInterface() = default;
    virtual grpc::Status GetRiskState(grpc::ClientContext* context,
                                      const RiskStateRequest& request,
                                      RiskState* response) = 0;
};

class GrpcMarketDataStreamReader final : public MarketDataStreamReader {
public:
    explicit GrpcMarketDataStreamReader(std::unique_ptr<grpc::ClientReader<StreamOhlcvUpdate>> reader)
        : m_reader(std::move(reader))
    {
    }

    bool Read(StreamOhlcvUpdate* update) override
    {
        if (!m_reader)
            return false;
        return m_reader->Read(update);
    }

    grpc::Status Finish() override
    {
        if (!m_reader)
            return grpc::Status::OK;
        return m_reader->Finish();
    }

private:
    std::unique_ptr<grpc::ClientReader<StreamOhlcvUpdate>> m_reader;
};

class GrpcMarketDataStub final : public TradingClient::MarketDataStubInterface {
public:
    explicit GrpcMarketDataStub(std::shared_ptr<grpc::Channel> channel)
        : m_stub(botcore::trading::v1::MarketDataService::NewStub(std::move(channel)))
    {
    }

    grpc::Status GetOhlcvHistory(grpc::ClientContext* context,
                                 const GetOhlcvHistoryRequest& request,
                                 GetOhlcvHistoryResponse* response) override
    {
        return m_stub->GetOhlcvHistory(context, request, response);
    }

    std::unique_ptr<MarketDataStreamReader> StreamOhlcv(
        grpc::ClientContext* context,
        const StreamOhlcvRequest& request) override
    {
        auto reader = m_stub->StreamOhlcv(context, request);
        if (!reader)
            return {};
        return std::make_unique<GrpcMarketDataStreamReader>(std::move(reader));
    }

    grpc::Status ListTradableInstruments(grpc::ClientContext* context,
                                         const ListTradableInstrumentsRequest& request,
                                         ListTradableInstrumentsResponse* response) override
    {
        return m_stub->ListTradableInstruments(context, request, response);
    }

private:
    std::unique_ptr<botcore::trading::v1::MarketDataService::Stub> m_stub;
};

class GrpcRiskServiceStub final : public TradingClient::RiskServiceStubInterface {
public:
    explicit GrpcRiskServiceStub(std::shared_ptr<grpc::Channel> channel)
        : m_stub(botcore::trading::v1::RiskService::NewStub(std::move(channel)))
    {
    }

    grpc::Status GetRiskState(grpc::ClientContext* context,
                              const RiskStateRequest& request,
                              RiskState* response) override
    {
        return m_stub->GetRiskState(context, request, response);
    }

private:
    std::unique_ptr<botcore::trading::v1::RiskService::Stub> m_stub;
};

google::protobuf::Timestamp toProtoTimestamp(qint64 timestampMs)
{
    google::protobuf::Timestamp ts;
    ts.set_seconds(timestampMs / 1000);
    ts.set_nanos(static_cast<int32_t>((timestampMs % 1000) * 1000000));
    return ts;
}

qint64 timestampToMs(const google::protobuf::Timestamp& ts)
{
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

class InProcessMarketDataStreamReader final : public MarketDataStreamReader {
public:
    InProcessMarketDataStreamReader(grpc::ClientContext* context,
                                    std::shared_ptr<std::vector<OhlcvCandle>> candles,
                                    bool loop)
        : m_context(context)
        , m_candles(std::move(candles))
        , m_loop(loop)
    {
    }

    bool Read(StreamOhlcvUpdate* update) override
    {
        if (!update)
            return false;

        if (m_context && m_context->IsCancelled())
            return false;

        if (!m_snapshotDelivered) {
            m_snapshotDelivered = true;
            auto* snapshot = update->mutable_snapshot();
            auto* series = snapshot->mutable_candles();
            series->Clear();
            for (const auto& candle : *m_candles) {
                auto* out = series->Add();
                *out = candle;
            }
            return true;
        }

        if (m_candles->empty())
            return false;

        if (m_index >= static_cast<int>(m_candles->size())) {
            if (!m_loop)
                return false;
            m_index = 0;
        }

        auto* increment = update->mutable_increment();
        increment->mutable_candle()->CopyFrom((*m_candles)[m_index]);
        ++m_index;
        for (int i = 0; i < 15; ++i) {
            if (m_context && m_context->IsCancelled())
                return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return true;
    }

    grpc::Status Finish() override { return grpc::Status::OK; }

private:
    grpc::ClientContext* m_context = nullptr;
    std::shared_ptr<std::vector<OhlcvCandle>> m_candles;
    bool m_snapshotDelivered = false;
    bool m_loop = false;
    int m_index = 0;
};

class InProcessMarketDataStub final : public TradingClient::MarketDataStubInterface {
public:
    InProcessMarketDataStub(const TradingClient::InstrumentConfig& config, const QString& datasetPath)
    {
        loadDataset(config, datasetPath);
    }

    grpc::Status GetOhlcvHistory(grpc::ClientContext*,
                                 const GetOhlcvHistoryRequest& request,
                                 GetOhlcvHistoryResponse* response) override
    {
        if (!response)
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "response null");
        response->clear_candles();
        const int limit = request.limit() > 0 ? request.limit() : static_cast<int>(m_candles->size());
        const int start = std::max(0, static_cast<int>(m_candles->size()) - limit);
        for (int i = start; i < static_cast<int>(m_candles->size()); ++i) {
            auto* out = response->add_candles();
            *out = (*m_candles)[i];
        }
        return grpc::Status::OK;
    }

    std::unique_ptr<MarketDataStreamReader> StreamOhlcv(grpc::ClientContext* context,
                                                        const StreamOhlcvRequest&) override
    {
        return std::make_unique<InProcessMarketDataStreamReader>(context, m_candles, true);
    }

    grpc::Status ListTradableInstruments(grpc::ClientContext*,
                                         const ListTradableInstrumentsRequest& request,
                                         ListTradableInstrumentsResponse* response) override
    {
        if (!response)
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "response null");
        response->clear_instruments();
        auto* listing = response->add_instruments();
        auto* instrument = listing->mutable_instrument();
        instrument->set_exchange(request.exchange());
        if (instrument->exchange().empty())
            instrument->set_exchange(m_config.exchange.toStdString());
        instrument->set_symbol(m_config.symbol.toStdString());
        instrument->set_venue_symbol(m_config.venueSymbol.toStdString());
        instrument->set_quote_currency(m_config.quoteCurrency.toStdString());
        instrument->set_base_currency(m_config.baseCurrency.toStdString());
        listing->set_price_step(0.01);
        listing->set_amount_step(0.001);
        listing->set_min_notional(5.0);
        listing->set_min_amount(0.001);
        listing->set_max_amount(25.0);
        listing->set_min_price(1.0);
        listing->set_max_price(100000.0);
        return grpc::Status::OK;
    }

    void updateInstrument(const TradingClient::InstrumentConfig& config)
    {
        m_config = config;
    }

private:
    void loadDataset(const TradingClient::InstrumentConfig& config, const QString& datasetPath)
    {
        m_config = config;
        QString path = datasetPath;
        if (path.trimmed().isEmpty())
            path = QStringLiteral("data/sample_ohlcv/trend.csv");
        path = bot::shell::utils::expandPath(path);

        auto candles = std::make_shared<std::vector<OhlcvCandle>>();
        QFile file(path);
        if (!file.exists() || !file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            buildSyntheticDataset(*candles);
            m_candles = candles;
            return;
        }

        QTextStream stream(&file);
        if (!stream.atEnd())
            stream.readLine();
        int sequence = 0;
        while (!stream.atEnd()) {
            const QString line = stream.readLine();
            const QStringList parts = line.split(QLatin1Char(','));
            if (parts.size() < 6)
                continue;
            const QString timestampRaw = parts.at(0).trimmed();
            QDateTime timestamp = QDateTime::fromString(timestampRaw, Qt::ISODate);
            if (!timestamp.isValid())
                timestamp = QDateTime::fromString(timestampRaw, QStringLiteral("yyyy-MM-dd HH:mm:ss"));
            if (!timestamp.isValid())
                continue;
            timestamp.setTimeSpec(Qt::UTC);

            bool ok = false;
            const double open = parts.at(1).toDouble(&ok);
            if (!ok)
                continue;
            const double high = parts.at(2).toDouble(&ok);
            if (!ok)
                continue;
            const double low = parts.at(3).toDouble(&ok);
            if (!ok)
                continue;
            const double close = parts.at(4).toDouble(&ok);
            if (!ok)
                continue;
            const double volume = parts.at(5).toDouble(&ok);
            if (!ok)
                continue;

            OhlcvCandle candle;
            *candle.mutable_open_time() = toProtoTimestamp(timestamp.toMSecsSinceEpoch());
            candle.set_open(open);
            candle.set_high(high);
            candle.set_low(low);
            candle.set_close(close);
            candle.set_volume(volume);
            candle.set_closed(true);
            candle.set_sequence(sequence++);
            candles->push_back(candle);
        }

        if (candles->empty())
            buildSyntheticDataset(*candles);
        m_candles = candles;
    }

    static void buildSyntheticDataset(std::vector<OhlcvCandle>& candles)
    {
        candles.clear();
        candles.reserve(64);
        const qint64 baseMs = QDateTime(QDate(2023, 1, 1), QTime(0, 0), Qt::UTC).toMSecsSinceEpoch();
        double price = 100.0;
        std::mt19937_64 rng{123456789ULL};
        std::normal_distribution<double> drift(0.05, 0.15);
        for (int i = 0; i < 64; ++i) {
            const double delta = drift(rng);
            const double open = price;
            const double close = std::max(1.0, open + delta);
            const double high = std::max(open, close) + std::abs(drift(rng));
            const double low = std::max(1.0, std::min(open, close) - std::abs(drift(rng)));
            const double volume = 900.0 + std::abs(drift(rng)) * 150.0;
            price = close;

            OhlcvCandle candle;
            *candle.mutable_open_time() = toProtoTimestamp(baseMs + static_cast<qint64>(i) * 60 * 1000);
            candle.set_open(open);
            candle.set_high(high);
            candle.set_low(low);
            candle.set_close(close);
            candle.set_volume(volume);
            candle.set_closed(true);
            candle.set_sequence(i);
            candles.push_back(candle);
        }
    }

    TradingClient::InstrumentConfig m_config;
    std::shared_ptr<std::vector<OhlcvCandle>> m_candles;
};

class InProcessRiskServiceStub final : public TradingClient::RiskServiceStubInterface {
public:
    grpc::Status GetRiskState(grpc::ClientContext*,
                              const RiskStateRequest&,
                              RiskState* response) override
    {
        if (!response)
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "response null");
        response->Clear();
        response->set_profile(botcore::trading::v1::RiskProfile::RISK_PROFILE_BALANCED);
        response->set_portfolio_value(25000.0);
        response->set_current_drawdown(0.015);
        response->set_max_daily_loss(0.05);
        response->set_used_leverage(1.5);
        auto* generated = response->mutable_generated_at();
        *generated = toProtoTimestamp(QDateTime::currentDateTimeUtc().toMSecsSinceEpoch());
        auto* limit = response->add_limits();
        limit->set_code("max_notional");
        limit->set_max_value(5000.0);
        limit->set_current_value(1200.0);
        limit->set_threshold_value(4500.0);
        auto* limit2 = response->add_limits();
        limit2->set_code("max_positions");
        limit2->set_max_value(10.0);
        limit2->set_current_value(3.0);
        limit2->set_threshold_value(8.0);
        return grpc::Status::OK;
    }
};

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

std::unique_ptr<TradingClient::IMarketDataTransport> TradingClient::makeTransport(TransportMode mode) const
{
    switch (mode) {
    case TransportMode::Grpc:
        return std::make_unique<GrpcMarketDataTransport>();
    case TransportMode::InProcess:
        return std::make_unique<InProcessMarketDataTransport>();
    }
    return {};
}

TradingClient::TradingClient(QObject* parent)
    : QObject(parent) {
    qRegisterMetaType<QList<OhlcvPoint>>("QList<OhlcvPoint>");
    qRegisterMetaType<PerformanceGuard>("PerformanceGuard");
    qRegisterMetaType<RiskSnapshotData>("RiskSnapshotData");
    qRegisterMetaType<IndicatorSample>("IndicatorSample");
    qRegisterMetaType<QVector<IndicatorSample>>("QVector<IndicatorSample>");
    qRegisterMetaType<SignalEventEntry>("SignalEventEntry");
    qRegisterMetaType<QVector<SignalEventEntry>>("QVector<SignalEventEntry>");
    qRegisterMetaType<MarketRegimeSnapshotEntry>("MarketRegimeSnapshotEntry");

    m_regimeClassifier = std::make_unique<MarketRegimeClassifierBridge>(this);

    m_transport = makeTransport(m_transportMode);
    if (m_transport) {
        m_transport->setInstrument(m_instrumentConfig);
        m_transport->setEndpoint(m_endpoint);
        m_transport->setTlsConfig(m_tlsConfig);
        m_transport->setDatasetPath(m_inProcessDatasetPath);
    }
}

TradingClient::~TradingClient() {
    stop();
}

void TradingClient::setEndpoint(const QString& endpoint) {
    if (endpoint == m_endpoint) {
        return;
    }
    m_endpoint = endpoint;
    if (m_transport) {
        m_transport->setEndpoint(m_endpoint);
        m_transport->ensureReady();
    }
}

void TradingClient::setTransportMode(TransportMode mode)
{
    if (mode == m_transportMode)
        return;
    const bool wasRunning = m_running.load();
    if (wasRunning)
        stop();
    m_transportMode = mode;
    m_transport = makeTransport(m_transportMode);
    if (m_transport) {
        m_transport->setInstrument(m_instrumentConfig);
        m_transport->setEndpoint(m_endpoint);
        m_transport->setTlsConfig(m_tlsConfig);
        m_transport->setDatasetPath(m_inProcessDatasetPath);
        m_transport->ensureReady();
    }
    if (wasRunning)
        start();
}

void TradingClient::setTransportMode(TransportMode mode)
{
    if (mode == m_transportMode)
        return;
    const bool wasRunning = m_running.load();
    if (wasRunning)
        stop();
    m_transportMode = mode;
    m_channel.reset();
    m_marketDataStub.reset();
    m_riskStub.reset();
    if (wasRunning)
        start();
}

void TradingClient::setInstrument(const InstrumentConfig& config) {
    m_instrumentConfig = config;
    if (m_transportMode == TransportMode::InProcess) {
        if (auto* inProcess = dynamic_cast<InProcessMarketDataStub*>(m_marketDataStub.get())) {
            inProcess->updateInstrument(m_instrumentConfig);
        }
    }
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
    if (m_transport) {
        m_transport->setTlsConfig(m_tlsConfig);
        m_transport->ensureReady();
    }
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

void TradingClient::setInProcessDatasetPath(const QString& path)
{
    if (m_inProcessDatasetPath == path)
        return;
    m_inProcessDatasetPath = path;
    if (m_transportMode == TransportMode::InProcess) {
        m_marketDataStub.reset();
        triggerStreamRestart();
    }
}

void TradingClient::setRegimeThresholdsPath(const QString& path)
{
    const QString trimmed = path.trimmed();
    if (m_regimeThresholdPath == trimmed) {
        reloadRegimeThresholds();
        return;
    }
    m_regimeThresholdPath = trimmed;
    reloadRegimeThresholds();
}

void TradingClient::reloadRegimeThresholds()
{
    if (!m_regimeClassifier)
        return;

    if (m_regimeThresholdPath.trimmed().isEmpty())
        return;

    if (!m_regimeClassifier->loadThresholdsFromFile(m_regimeThresholdPath)) {
        qCWarning(lcTradingClient)
            << "Nie udało się wczytać progów MarketRegimeClassifier z" << m_regimeThresholdPath;
    }
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

bool TradingClient::hasGrpcChannelForTesting() const
{
    return static_cast<bool>(m_channel);
}

void TradingClient::start() {
    if (m_running.exchange(true)) {
        return;
    }
    if (m_streamThread.joinable()) {
        m_streamThread.join();
    }
    m_restartRequested.store(false);
    ensureTransport();

    if (!m_marketDataStub) {
        if (m_transportMode == TransportMode::InProcess) {
            qCWarning(lcTradingClient)
                << "Brak poprawnie zainicjalizowanego transportu in-process (dataset:" << m_inProcessDatasetPath
                << ')';
        } else {
            qCWarning(lcTradingClient)
                << "Brak poprawnie zainicjalizowanego stubu MarketDataService dla endpointu" << m_endpoint;
        }
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
    const grpc::Status historyStatus =
        m_transport->getOhlcvHistory(historyContext.get(), historyReq, &historyResp);
    if (historyStatus.ok()) {
        const QList<OhlcvPoint> history = convertHistory(historyResp.candles());
        QVector<IndicatorSample> emaFast;
        QVector<IndicatorSample> emaSlow;
        QVector<IndicatorSample> vwap;
        QVector<SignalEventEntry> signalHistory;
        std::optional<MarketRegimeSnapshotEntry> regime;
        {
            std::lock_guard<std::mutex> lock(m_historyMutex);
            m_cachedHistory = QVector<OhlcvPoint>::fromList(history);
            std::sort(m_cachedHistory.begin(), m_cachedHistory.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.timestampMs < rhs.timestampMs;
            });
            if (m_historyLimit > 0 && m_cachedHistory.size() > m_historyLimit) {
                m_cachedHistory.erase(m_cachedHistory.begin(),
                                      m_cachedHistory.begin() + (m_cachedHistory.size() - m_historyLimit));
            }
            rebuildIndicatorSnapshots(m_cachedHistory);
            emaFast = m_cachedEmaFast;
            emaSlow = m_cachedEmaSlow;
            vwap = m_cachedVwap;
            computeSignalHistory(m_cachedEmaFast, m_cachedEmaSlow, m_cachedVwap, m_signalHistory);
            signalHistory = m_signalHistory;
            regime = evaluateRegimeSnapshotLocked();
        }

        Q_EMIT historyReceived(history);
        Q_EMIT indicatorSnapshotReceived(QStringLiteral("ema_fast"), emaFast);
        Q_EMIT indicatorSnapshotReceived(QStringLiteral("ema_slow"), emaSlow);
        Q_EMIT indicatorSnapshotReceived(QStringLiteral("vwap"), vwap);
        if (!signalHistory.isEmpty()) {
            Q_EMIT signalHistoryReceived(signalHistory);
        }
        if (regime.has_value()) {
            Q_EMIT marketRegimeUpdated(*regime);
        }
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
    if (m_transport) {
        m_transport->shutdown();
    }
    if (wasRunning) {
        Q_EMIT streamingChanged();
        Q_EMIT connectionStateChanged(tr("stopped"));
    }
}

void TradingClient::ensureStub() {
    if (m_transportMode == TransportMode::InProcess) {
        if (!m_marketDataStub) {
            m_marketDataStub = std::make_unique<InProcessMarketDataStub>(m_instrumentConfig,
                                                                         m_inProcessDatasetPath);
        } else if (auto* inProcess = dynamic_cast<InProcessMarketDataStub*>(m_marketDataStub.get())) {
            inProcess->updateInstrument(m_instrumentConfig);
        }
        if (!m_riskStub) {
            m_riskStub = std::make_unique<InProcessRiskServiceStub>();
        }
        m_channel.reset();
        return;
    }

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
        m_marketDataStub = std::make_unique<GrpcMarketDataStub>(m_channel);
    }
    if (m_channel && !m_riskStub) {
        m_riskStub = std::make_unique<GrpcRiskServiceStub>(m_channel);
    }
    m_transport->ensureReady();
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
        ensureTransport();
        if (!m_transport || !m_transport->hasConnectivity()) {
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
        if (!reader) {
            QMetaObject::invokeMethod(
                this,
                [this]() { Q_EMIT connectionStateChanged(tr("stream unavailable")); },
                Qt::QueuedConnection);
            ++attempt;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            continue;
        }
        StreamOhlcvUpdate update;
        bool receivedAny = false;

        while (m_running.load() && reader->Read(&update)) {
            receivedAny = true;
            attempt = 0;
            if (update.has_snapshot()) {
                const QList<OhlcvPoint> history = convertHistory(update.snapshot().candles());
                QVector<IndicatorSample> emaFast;
                QVector<IndicatorSample> emaSlow;
                QVector<IndicatorSample> vwap;
                QVector<SignalEventEntry> signalHistory;
                std::optional<MarketRegimeSnapshotEntry> regime;
                {
                    std::lock_guard<std::mutex> lock(m_historyMutex);
                    m_cachedHistory = QVector<OhlcvPoint>::fromList(history);
                    std::sort(m_cachedHistory.begin(), m_cachedHistory.end(), [](const auto& lhs, const auto& rhs) {
                        return lhs.timestampMs < rhs.timestampMs;
                    });
                    rebuildIndicatorSnapshots(m_cachedHistory);
                    emaFast = m_cachedEmaFast;
                    emaSlow = m_cachedEmaSlow;
                    vwap = m_cachedVwap;
                    computeSignalHistory(m_cachedEmaFast, m_cachedEmaSlow, m_cachedVwap, m_signalHistory);
                    signalHistory = m_signalHistory;
                    regime = evaluateRegimeSnapshotLocked();
                }
                QMetaObject::invokeMethod(
                    this,
                    [this,
                     history,
                     emaFast = std::move(emaFast),
                     emaSlow = std::move(emaSlow),
                     vwap = std::move(vwap),
                     signalHistory = std::move(signalHistory),
                     regime]() mutable {
                        Q_EMIT historyReceived(history);
                        Q_EMIT indicatorSnapshotReceived(QStringLiteral("ema_fast"), emaFast);
                        Q_EMIT indicatorSnapshotReceived(QStringLiteral("ema_slow"), emaSlow);
                        Q_EMIT indicatorSnapshotReceived(QStringLiteral("vwap"), vwap);
                        if (!signalHistory.isEmpty()) {
                            Q_EMIT signalHistoryReceived(signalHistory);
                        }
                        if (regime.has_value()) {
                            Q_EMIT marketRegimeUpdated(*regime);
                        }
                    },
                    Qt::QueuedConnection);
            }
            if (update.has_increment()) {
                const OhlcvPoint point = convertCandle(update.increment().candle());
                IndicatorSample fastSample;
                IndicatorSample slowSample;
                IndicatorSample vwapSample;
                std::optional<SignalEventEntry> latestSignal;
                std::optional<MarketRegimeSnapshotEntry> regime;
                {
                    std::lock_guard<std::mutex> lock(m_historyMutex);
                    if (!m_cachedHistory.isEmpty()) {
                        auto& last = m_cachedHistory.last();
                        if (last.sequence == point.sequence || last.timestampMs == point.timestampMs) {
                            last = point;
                        } else {
                            m_cachedHistory.append(point);
                            if (m_cachedHistory.size() >= 2 && m_cachedHistory[m_cachedHistory.size() - 2].timestampMs > point.timestampMs) {
                                std::sort(m_cachedHistory.begin(), m_cachedHistory.end(), [](const auto& lhs, const auto& rhs) {
                                    return lhs.timestampMs < rhs.timestampMs;
                                });
                            }
                        }
                    } else {
                        m_cachedHistory.append(point);
                    }
                    if (m_historyLimit > 0 && m_cachedHistory.size() > m_historyLimit) {
                        const int removeCount = m_cachedHistory.size() - m_historyLimit;
                        m_cachedHistory.erase(m_cachedHistory.begin(), m_cachedHistory.begin() + removeCount);
                    }
                    const qsizetype previousSignals = m_signalHistory.size();
                    rebuildIndicatorSnapshots(m_cachedHistory);
                    computeSignalHistory(m_cachedEmaFast, m_cachedEmaSlow, m_cachedVwap, m_signalHistory);
                    if (!m_cachedEmaFast.isEmpty()) {
                        fastSample = m_cachedEmaFast.last();
                    }
                    if (!m_cachedEmaSlow.isEmpty()) {
                        slowSample = m_cachedEmaSlow.last();
                    }
                    if (!m_cachedVwap.isEmpty()) {
                        vwapSample = m_cachedVwap.last();
                    }
                    if (m_signalHistory.size() > previousSignals && !m_signalHistory.isEmpty()) {
                        latestSignal = m_signalHistory.last();
                    }
                    regime = evaluateRegimeSnapshotLocked();
                }
                QMetaObject::invokeMethod(
                    this,
                    [this, point, fastSample, slowSample, vwapSample, latestSignal, regime]() {
                        Q_EMIT candleReceived(point);
                        if (fastSample.timestampMs != 0) {
                            Q_EMIT indicatorSampleReceived(fastSample);
                        }
                        if (slowSample.timestampMs != 0) {
                            Q_EMIT indicatorSampleReceived(slowSample);
                        }
                        if (vwapSample.timestampMs != 0) {
                            Q_EMIT indicatorSampleReceived(vwapSample);
                        }
                        if (latestSignal.has_value()) {
                            Q_EMIT signalEventReceived(*latestSignal);
                        }
                        if (regime.has_value()) {
                            Q_EMIT marketRegimeUpdated(*regime);
                        }
                    },
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
        if (m_transportMode == TransportMode::InProcess) {
            qCWarning(lcTradingClient) << "Brak transportu in-process dla RiskService – pomijam odczyt stanu ryzyka.";
        } else {
            qCWarning(lcTradingClient) << "Brak stubu RiskService – pomijam odczyt stanu ryzyka.";
        }
        return;
    }
    auto riskContext = createContext();
    RiskStateRequest request;
    RiskState response;
    const grpc::Status status = m_transport->getRiskState(riskContext.get(), request, &response);
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

QVector<TradingClient::TradableInstrument> TradingClient::listTradableInstruments(const QString& exchange)
{
    QVector<TradableInstrument> instruments;
    ensureStub();
    if (!m_marketDataStub) {
        if (m_transportMode == TransportMode::InProcess) {
            qCWarning(lcTradingClient)
                << "ListTradableInstruments pominięte – brak transportu in-process (dataset:" << m_inProcessDatasetPath
                << ')';
        } else {
            qCWarning(lcTradingClient)
                << "ListTradableInstruments pominięte – brak połączenia z MarketDataService dla" << m_endpoint;
        }
        return instruments;
    }

    const QString normalizedExchange = exchange.trimmed().toUpper();
    if (normalizedExchange.isEmpty()) {
        return instruments;
    }

    grpc::ClientContext context;
    applyAuthMetadata(context);
    ListTradableInstrumentsRequest request;
    request.set_exchange(normalizedExchange.toStdString());
    ListTradableInstrumentsResponse response;

    const grpc::Status status = m_transport->listTradableInstruments(&context, request, &response);
    if (!status.ok()) {
        qCWarning(lcTradingClient)
            << "ListTradableInstruments nie powiodło się:" << QString::fromStdString(status.error_message());
        return instruments;
    }

    instruments.reserve(static_cast<int>(response.instruments_size()));
    for (const auto& item : response.instruments()) {
        TradableInstrument listing;
        listing.config.exchange = QString::fromStdString(item.instrument().exchange());
        if (listing.config.exchange.isEmpty()) {
            listing.config.exchange = normalizedExchange;
        }
        listing.config.symbol = QString::fromStdString(item.instrument().symbol());
        listing.config.venueSymbol = QString::fromStdString(item.instrument().venue_symbol());
        listing.config.quoteCurrency = QString::fromStdString(item.instrument().quote_currency());
        listing.config.baseCurrency = QString::fromStdString(item.instrument().base_currency());
        listing.config.granularityIso8601 = m_instrumentConfig.granularityIso8601;
        listing.priceStep = item.price_step();
        listing.amountStep = item.amount_step();
        listing.minNotional = item.min_notional();
        listing.minAmount = item.min_amount();
        listing.maxAmount = item.max_amount();
        listing.minPrice = item.min_price();
        listing.maxPrice = item.max_price();
        instruments.append(listing);
    }
    return instruments;
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
    if (m_transport) {
        m_transport->requestRestart();
    }
    m_restartRequested.store(true);
    std::lock_guard<std::mutex> lock(m_contextMutex);
    if (m_activeContext) {
        m_activeContext->TryCancel();
    }
}

void TradingClient::rebuildIndicatorSnapshots(const QVector<OhlcvPoint>& history)
{
    m_cachedEmaFast.clear();
    m_cachedEmaSlow.clear();
    m_cachedVwap.clear();

    m_cachedEmaFast.reserve(history.size());
    m_cachedEmaSlow.reserve(history.size());
    m_cachedVwap.reserve(history.size());

    if (history.isEmpty()) {
        return;
    }

    const int fastPeriod = 12;
    const int slowPeriod = 26;
    const double fastMultiplier = 2.0 / (fastPeriod + 1.0);
    const double slowMultiplier = 2.0 / (slowPeriod + 1.0);

    double emaFast = history.first().close;
    double emaSlow = history.first().close;
    double cumulativePv = 0.0;
    double cumulativeVolume = 0.0;

    for (int i = 0; i < history.size(); ++i) {
        const auto& candle = history.at(i);
        const double price = candle.close;

        if (i == 0) {
            emaFast = price;
            emaSlow = price;
        } else {
            emaFast = (price - emaFast) * fastMultiplier + emaFast;
            emaSlow = (price - emaSlow) * slowMultiplier + emaSlow;
        }

        IndicatorSample fastSample{QStringLiteral("ema_fast"), candle.timestampMs, emaFast};
        IndicatorSample slowSample{QStringLiteral("ema_slow"), candle.timestampMs, emaSlow};
        m_cachedEmaFast.append(fastSample);
        m_cachedEmaSlow.append(slowSample);

        cumulativePv += price * candle.volume;
        cumulativeVolume += candle.volume;
        double vwapValue = price;
        if (cumulativeVolume > 0.0) {
            vwapValue = cumulativePv / cumulativeVolume;
        }
        IndicatorSample vwapSample{QStringLiteral("vwap"), candle.timestampMs, vwapValue};
        m_cachedVwap.append(vwapSample);
    }
}

void TradingClient::computeSignalHistory(const QVector<IndicatorSample>& fast,
                                         const QVector<IndicatorSample>& slow,
                                         const QVector<IndicatorSample>& vwap,
                                         QVector<SignalEventEntry>& signalHistory)
{
    signalHistory.clear();
    const int count = std::min({fast.size(), slow.size(), vwap.size(), m_cachedHistory.size()});
    if (count == 0) {
        return;
    }

    double prevDiff = 0.0;
    bool   havePrevDiff = false;
    double prevPrice = m_cachedHistory.first().close;
    double prevVwap = vwap.first().value;

    for (int i = 0; i < count; ++i) {
        const double diff = fast.at(i).value - slow.at(i).value;
        if (havePrevDiff && diff * prevDiff < 0.0) {
            SignalEventEntry event;
            event.timestampMs = fast.at(i).timestampMs;
            if (diff > 0.0) {
                event.code = QStringLiteral("ema_bullish_cross");
                event.description = tr("Szybka EMA przecina wolną w górę");
            } else {
                event.code = QStringLiteral("ema_bearish_cross");
                event.description = tr("Szybka EMA przecina wolną w dół");
            }
            event.confidence = qBound(0.0, std::abs(diff) / std::max(0.0001, std::abs(slow.at(i).value)), 1.0);
            event.regime = m_lastRegimeSnapshot.regime;
            signalHistory.append(event);
        }
        prevDiff = diff;
        havePrevDiff = true;

        const double price = m_cachedHistory.at(i).close;
        const double vwapValue = vwap.at(i).value;
        const bool crossedUp = (prevPrice <= prevVwap) && (price > vwapValue);
        const bool crossedDown = (prevPrice >= prevVwap) && (price < vwapValue);
        if ((crossedUp || crossedDown) && i > 0) {
            SignalEventEntry event;
            event.timestampMs = m_cachedHistory.at(i).timestampMs;
            if (crossedUp) {
                event.code = QStringLiteral("vwap_breakout");
                event.description = tr("Cena wybija powyżej VWAP");
            } else {
                event.code = QStringLiteral("vwap_breakdown");
                event.description = tr("Cena spada poniżej VWAP");
            }
            event.confidence = qBound(0.0, std::abs(price - vwapValue) / std::max(0.0001, std::abs(vwapValue)), 1.0);
            event.regime = m_lastRegimeSnapshot.regime;
            signalHistory.append(event);
        }
        prevPrice = price;
        prevVwap = vwapValue;
    }

    std::sort(signalHistory.begin(), signalHistory.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.timestampMs < rhs.timestampMs;
    });
}

std::optional<MarketRegimeSnapshotEntry> TradingClient::evaluateRegimeSnapshotLocked()
{
    if (!m_regimeClassifier) {
        return std::nullopt;
    }
    const auto assessment = m_regimeClassifier->classify(m_cachedHistory);
    if (!assessment.has_value()) {
        return std::nullopt;
    }

    const bool sameTimestamp = m_lastRegimeSnapshot.timestampMs == assessment->timestampMs;
    const bool sameRegime = m_lastRegimeSnapshot.regime == assessment->regime;
    const bool sameTrend = qFuzzyCompare(1.0 + m_lastRegimeSnapshot.trendConfidence, 1.0 + assessment->trendConfidence);
    const bool sameMr = qFuzzyCompare(1.0 + m_lastRegimeSnapshot.meanReversionConfidence,
                                      1.0 + assessment->meanReversionConfidence);
    const bool sameDaily = qFuzzyCompare(1.0 + m_lastRegimeSnapshot.dailyConfidence, 1.0 + assessment->dailyConfidence);

    m_lastRegimeSnapshot = *assessment;
    if (sameTimestamp && sameRegime && sameTrend && sameMr && sameDaily) {
        return std::nullopt;
    }
    return assessment;
}

TradingClient::PreLiveChecklistResult TradingClient::runPreLiveChecklist() const {
    PreLiveChecklistResult result;

    if (m_transportMode == TransportMode::InProcess) {
        const QString datasetPath = bot::shell::utils::expandPath(m_inProcessDatasetPath);
        if (m_inProcessDatasetPath.trimmed().isEmpty()) {
            result.warnings.append(tr("Brak wskazanego pliku dataset – użyty zostanie syntetyczny feed."));
        } else if (!QFile::exists(datasetPath)) {
            result.warnings.append(tr("Dataset in-process nie istnieje: %1 – użyty zostanie syntetyczny feed.").arg(datasetPath));
        }
        result.ok = result.errors.isEmpty();
        return result;
    }

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
