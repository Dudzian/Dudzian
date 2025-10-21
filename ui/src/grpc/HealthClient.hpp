#pragma once

#include <QByteArray>
#include <QDateTime>
#include <QPair>
#include <QString>
#include <QStringList>
#include <QVector>

#include <memory>
#include <mutex>

#include "GrpcTlsConfig.hpp"

namespace botcore::trading::v1 {
class HealthService;
class HealthCheckResponse;
} // namespace botcore::trading::v1

namespace grpc {
class Channel;
class ClientContext;
} // namespace grpc

class HealthClientInterface {
public:
    struct HealthCheckResult {
        bool       ok = false;
        QString    version;
        QString    gitCommit;
        QDateTime  startedAtUtc;
        QString    errorMessage;
    };

    virtual ~HealthClientInterface() = default;

    virtual void setEndpoint(const QString& endpoint) = 0;
    virtual void setTlsConfig(const GrpcTlsConfig& config) = 0;
    virtual void setAuthToken(const QString& token) = 0;
    virtual void setRbacRole(const QString& role) = 0;
    virtual void setRbacScopes(const QStringList& scopes) = 0;

    virtual QVector<QPair<QByteArray, QByteArray>> authMetadataForTesting() const = 0;
    virtual HealthCheckResult check() = 0;
};

class HealthClient final : public HealthClientInterface {
public:
    HealthClient();
    ~HealthClient() override;

    void setEndpoint(const QString& endpoint) override;
    void setTlsConfig(const GrpcTlsConfig& config) override;
    void setAuthToken(const QString& token) override;
    void setRbacRole(const QString& role) override;
    void setRbacScopes(const QStringList& scopes) override;

    QVector<QPair<QByteArray, QByteArray>> authMetadataForTesting() const override;
    HealthCheckResult check() override;

    bool hasChannelForTesting() const;
    bool hasStubForTesting() const;

    struct PreflightResult {
        bool        ok = false;
        QStringList errors;
        QStringList warnings;
    };

    PreflightResult runPreflightChecklist() const;

private:
    void ensureStub();
    void resetChannelLocked();
    std::unique_ptr<grpc::ClientContext> buildContext() const;
    static QDateTime convertTimestamp(const botcore::trading::v1::HealthCheckResponse& response);

    mutable std::mutex m_mutex;
    QString            m_endpoint;
    GrpcTlsConfig      m_tlsConfig;
    QString            m_authToken;
    QString            m_rbacRole;
    QStringList        m_rbacScopes;

    std::shared_ptr<grpc::Channel>                                    m_channel;
    std::unique_ptr<botcore::trading::v1::HealthService::Stub>        m_stub;
};

