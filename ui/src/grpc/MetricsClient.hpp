#pragma once

#include <QByteArray>
#include <QPair>
#include <QString>
#include <QStringList>
#include <QVector>
#include <memory>
#include <mutex>
#include <optional>

#include "telemetry/TelemetryTlsConfig.hpp"

namespace grpc {
class Channel;
class ClientContext;
} // namespace grpc

namespace botcore::trading::v1 {
class MetricsService;
class MetricsSnapshot;
class MetricsAck;
} // namespace botcore::trading::v1

struct MetricsPreflightResult {
    bool ok = false;
    QStringList warnings;
    QStringList errors;
};

class MetricsClientInterface {
public:
    virtual ~MetricsClientInterface() = default;

    virtual void setEndpoint(const QString& endpoint) = 0;
    virtual void setTlsConfig(const TelemetryTlsConfig& config) = 0;
    virtual void setAuthToken(const QString& token) = 0;
    virtual void setRbacRole(const QString& role) = 0;

    virtual bool pushSnapshot(const botcore::trading::v1::MetricsSnapshot& snapshot,
                              QString* errorMessage = nullptr) = 0;
};

class MetricsClient final : public MetricsClientInterface {
public:
    MetricsClient();
    ~MetricsClient();

    void setEndpoint(const QString& endpoint) override;
    void setTlsConfig(const TelemetryTlsConfig& config) override;
    void setAuthToken(const QString& token) override;
    void setRbacRole(const QString& role) override;

    bool pushSnapshot(const botcore::trading::v1::MetricsSnapshot& snapshot,
                      QString* errorMessage = nullptr) override;

    MetricsPreflightResult runPreflightChecklist() const;
    QVector<QPair<QByteArray, QByteArray>> authMetadataForTesting() const;

private:
    void ensureChannel();
    std::unique_ptr<grpc::ClientContext> buildContext() const;

    QString m_endpoint;
    TelemetryTlsConfig m_tlsConfig{};
    QString m_authToken;
    QString m_rbacRole;

    mutable std::mutex m_mutex;
    std::shared_ptr<grpc::Channel> m_channel;
    std::unique_ptr<botcore::trading::v1::MetricsService::Stub> m_stub;
};

