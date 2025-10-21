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
} // namespace botcore::trading::v1

namespace grpc {
class Channel;
class ClientContext;
} // namespace grpc

class TradingClient : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool streaming READ isStreaming NOTIFY streamingChanged)

public:
    struct InstrumentConfig {
        QString exchange;
        QString symbol;
        QString venueSymbol;
        QString quoteCurrency;
        QString baseCurrency;
        QString granularityIso8601;
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
    void setInstrument(const InstrumentConfig& config);
    void setHistoryLimit(int limit);
    void setPerformanceGuard(const PerformanceGuard& guard);
    void setTlsConfig(const TlsConfig& config);
    void setAuthToken(const QString& token);
    void setRbacRole(const QString& role);
    void setRbacScopes(const QStringList& scopes);

    QVector<QPair<QByteArray, QByteArray>> authMetadataForTesting() const;

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
    void performanceGuardUpdated(const PerformanceGuard& guard);
    void connectionStateChanged(const QString& state);
    void riskStateReceived(const RiskSnapshotData& snapshot);

private:
    void ensureStub();
    QList<OhlcvPoint> convertHistory(
        const google::protobuf::RepeatedPtrField<botcore::trading::v1::OhlcvCandle>& candles) const;
    OhlcvPoint convertCandle(const botcore::trading::v1::OhlcvCandle& candle) const;
    void streamLoop();
    RiskSnapshotData convertRiskState(const botcore::trading::v1::RiskState& state) const;
    std::shared_ptr<grpc::ClientContext> createContext() const;
    void applyAuthMetadata(grpc::ClientContext& context) const;
    std::vector<std::pair<std::string, std::string>> buildAuthMetadata() const;
    void triggerStreamRestart();

    // --- Konfiguracja połączenia/rynku ---
    QString m_endpoint = QStringLiteral("127.0.0.1:50061");
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

    // --- gRPC ---
    std::shared_ptr<grpc::Channel> m_channel;
    std::unique_ptr<botcore::trading::v1::MarketDataService::Stub> m_marketDataStub;
    std::unique_ptr<botcore::trading::v1::RiskService::Stub> m_riskStub;

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
};
