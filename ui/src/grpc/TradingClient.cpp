#include "TradingClient.hpp"

#include <QMetaObject>
#include <QtGlobal>

#include <google/protobuf/timestamp.pb.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/security/credentials.h>

#include "trading.grpc.pb.h"

using botcore::trading::v1::GetOhlcvHistoryRequest;
using botcore::trading::v1::GetOhlcvHistoryResponse;
using botcore::trading::v1::Instrument;
using botcore::trading::v1::MarketDataService;
using botcore::trading::v1::OhlcvCandle;
using botcore::trading::v1::StreamOhlcvRequest;
using botcore::trading::v1::StreamOhlcvUpdate;

namespace {
qint64 timestampToMs(const google::protobuf::Timestamp& ts) {
    return static_cast<qint64>(ts.seconds()) * 1000 + ts.nanos() / 1000000;
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
}

TradingClient::~TradingClient() {
    stop();
}

void TradingClient::setEndpoint(const QString& endpoint) {
    if (endpoint == m_endpoint) {
        return;
    }
    m_endpoint = endpoint;
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
        Q_EMIT connectionStateChanged(QStringLiteral("history error: %1").arg(QString::fromStdString(historyStatus.error_message())));
    }

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
    m_channel = grpc::CreateChannel(m_endpoint.toStdString(), grpc::InsecureChannelCredentials());
    m_marketDataStub = MarketDataService::NewStub(m_channel);
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
