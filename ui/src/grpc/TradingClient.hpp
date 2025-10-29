#pragma once

#include <QObject>
#include <QByteArray>
#include <QPair>
#include <QString>
#include <QStringList>
#include <QVector>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <google/protobuf/repeated_field.h>

#include "models/MarketDataStreams.hpp"
#include "models/OhlcvListModel.hpp"
#include "models/RiskTypes.hpp"
#include "utils/PerformanceGuard.hpp"
#include "GrpcTlsConfig.hpp"

namespace botcore::trading::v1 {
class Instrument;
class CandleGranularity;
class MarketDataService;
class OhlcvCandle;
class RiskService;
class RiskState;
class GetOhlcvHistoryRequest;
class GetOhlcvHistoryResponse;
class StreamOhlcvRequest;
class StreamOhlcvUpdate;
class RiskStateRequest;
class ListTradableInstrumentsRequest;
class ListTradableInstrumentsResponse;
} // namespace botcore::trading::v1

namespace grpc {
class Channel;
class ClientContext;
class Status;
} // namespace grpc

class TradingClient : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool streaming READ isStreaming NOTIFY streamingChanged)

public:
    enum class TransportMode {
        Grpc,
        InProcess,
    };

    struct InstrumentConfig {
        QString exchange;
        QString symbol;
        QString venueSymbol;
        QString quoteCurrency;
        QString baseCurrency;
        QString granularityIso8601;
    };

    struct TradableInstrument {
        InstrumentConfig config;
        double priceStep = 0.0;
        double amountStep = 0.0;
        double minNotional = 0.0;
        double minAmount = 0.0;
        double maxAmount = 0.0;
        double minPrice = 0.0;
        double maxPrice = 0.0;
    };

    using TlsConfig = GrpcTlsConfig;

    struct PreLiveChecklistResult {
        bool ok = false;
        QStringList warnings;
        QStringList errors;
    };

    explicit TradingClient(QObject* parent = nullptr);
    ~TradingClient() override;

    void setEndpoint(const QString& endpoint);
    void setTransportMode(TransportMode mode);
    void setInstrument(const InstrumentConfig& config);
    void setHistoryLimit(int limit);
    void setPerformanceGuard(const PerformanceGuard& guard);
    void setTlsConfig(const TlsConfig& config);
    void setAuthToken(const QString& token);
    void setRbacRole(const QString& role);
    void setRbacScopes(const QStringList& scopes);
    void setRegimeThresholdsPath(const QString& path);
    void reloadRegimeThresholds();
    void setInProcessDatasetPath(const QString& path);

    QVector<TradableInstrument> listTradableInstruments(const QString& exchange);

    QVector<QPair<QByteArray, QByteArray>> authMetadataForTesting() const;
    bool hasGrpcChannelForTesting() const;
    TransportMode transportMode() const { return m_transportMode; }

    // Używane przez Application.cpp
    InstrumentConfig instrumentConfig() const { return m_instrumentConfig; }
    TlsConfig tlsConfig() const { return m_tlsConfig; }

    bool isStreaming() const { return m_running.load(); }

    PreLiveChecklistResult runPreLiveChecklist() const;

public slots:
    void start();
    void stop();
    void refreshRiskState();

signals:
    void streamingChanged();
    void historyReceived(const QList<OhlcvPoint>& history);
    void candleReceived(const OhlcvPoint& candle);
    void indicatorSnapshotReceived(const QString& id, const QVector<IndicatorSample>& samples);
    void indicatorSampleReceived(const IndicatorSample& sample);
    void signalHistoryReceived(const QVector<SignalEventEntry>& events);
    void signalEventReceived(const SignalEventEntry& event);
    void marketRegimeUpdated(const MarketRegimeSnapshotEntry& snapshot);
    void performanceGuardUpdated(const PerformanceGuard& guard);
    void connectionStateChanged(const QString& state);
    void riskStateReceived(const RiskSnapshotData& snapshot);

private:
    class MarketDataStreamReader;
    class IMarketDataTransport {
    public:
        virtual ~IMarketDataTransport() = default;
        virtual void setInstrument(const InstrumentConfig& config) = 0;
        virtual void setEndpoint(const QString& endpoint) = 0;
        virtual void setTlsConfig(const TlsConfig& config) = 0;
        virtual void setDatasetPath(const QString& path) = 0;
        virtual bool ensureReady() = 0;
        virtual grpc::Status getOhlcvHistory(grpc::ClientContext* context,
                                             const botcore::trading::v1::GetOhlcvHistoryRequest& request,
                                             botcore::trading::v1::GetOhlcvHistoryResponse* response) = 0;
        virtual std::unique_ptr<class MarketDataStreamReader> streamOhlcv(
            grpc::ClientContext* context,
            const botcore::trading::v1::StreamOhlcvRequest& request) = 0;
        virtual grpc::Status getRiskState(grpc::ClientContext* context,
                                          const botcore::trading::v1::RiskStateRequest& request,
                                          botcore::trading::v1::RiskState* response) = 0;
        virtual grpc::Status listTradableInstruments(
            grpc::ClientContext* context,
            const botcore::trading::v1::ListTradableInstrumentsRequest& request,
            botcore::trading::v1::ListTradableInstrumentsResponse* response) = 0;
        virtual void shutdown() = 0;
        virtual void requestRestart() = 0;
        virtual bool hasConnectivity() const = 0;
        virtual bool isGrpcTransport() const = 0;
        virtual bool hasNativeChannel() const = 0;
    };

    void ensureTransport();
    std::unique_ptr<IMarketDataTransport> makeTransport(TransportMode mode) const;
    QList<OhlcvPoint> convertHistory(
        const google::protobuf::RepeatedPtrField<botcore::trading::v1::OhlcvCandle>& candles) const;
    OhlcvPoint convertCandle(const botcore::trading::v1::OhlcvCandle& candle) const;
    void streamLoop();
    RiskSnapshotData convertRiskState(const botcore::trading::v1::RiskState& state) const;
    std::shared_ptr<grpc::ClientContext> createContext() const;
    void applyAuthMetadata(grpc::ClientContext& context) const;
    std::vector<std::pair<std::string, std::string>> buildAuthMetadata() const;
    void triggerStreamRestart();
    void rebuildIndicatorSnapshots(const QVector<OhlcvPoint>& history);
    void computeSignalHistory(const QVector<IndicatorSample>& fast,
                              const QVector<IndicatorSample>& slow,
                              const QVector<IndicatorSample>& vwap,
                              QVector<SignalEventEntry>& signalHistory);
    std::optional<MarketRegimeSnapshotEntry> evaluateRegimeSnapshotLocked();

    // --- Konfiguracja połączenia/rynku ---
    QString m_endpoint = QStringLiteral("127.0.0.1:50061");
    TransportMode m_transportMode = TransportMode::Grpc;
    InstrumentConfig m_instrumentConfig{
        QStringLiteral("BINANCE"),
        QStringLiteral("BTC/USDT"),
        QStringLiteral("BTCUSDT"),
        QStringLiteral("USDT"),
        QStringLiteral("BTC"),
        QStringLiteral("PT1M")
    };
    PerformanceGuard m_guard{};
    int m_historyLimit = 500;
    TlsConfig m_tlsConfig{};
    QString   m_regimeThresholdPath;
    QString   m_inProcessDatasetPath;

    // --- gRPC ---
    std::shared_ptr<grpc::Channel> m_channel;
    class MarketDataStubInterface;
    class RiskServiceStubInterface;
    std::unique_ptr<MarketDataStubInterface> m_marketDataStub;
    std::unique_ptr<RiskServiceStubInterface> m_riskStub;

    // --- Streaming ---
    std::atomic<bool> m_running{false};
    std::thread m_streamThread;
    std::mutex m_contextMutex;
    std::shared_ptr<grpc::ClientContext> m_activeContext;
    std::atomic<bool> m_restartRequested{false};

    // --- Autoryzacja ---
    mutable std::mutex m_authMutex;
    QString m_authToken;
    QString m_rbacRole;
    QStringList m_rbacScopes;

    // --- Market data cache ---
    mutable std::mutex m_historyMutex;
    QVector<OhlcvPoint> m_cachedHistory;
    QVector<IndicatorSample> m_cachedEmaFast;
    QVector<IndicatorSample> m_cachedEmaSlow;
    QVector<IndicatorSample> m_cachedVwap;
    QVector<SignalEventEntry> m_signalHistory;
    MarketRegimeSnapshotEntry m_lastRegimeSnapshot;

    std::unique_ptr<class MarketRegimeClassifierBridge> m_regimeClassifier;
};
