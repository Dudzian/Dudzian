#include "HealthClient.hpp"

#include <QCryptographicHash>
#include <QFile>
#include <QFileInfo>
#include <QLoggingCategory>
#include <QIODevice>
#include <QSet>
#include <QSslCertificate>

#include <google/protobuf/empty.pb.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <optional>

#include "trading.grpc.pb.h"

Q_LOGGING_CATEGORY(lcHealthClient, "bot.shell.health.grpc")

namespace {

QString normalizeFingerprint(QString value)
{
    QString normalized = value.trimmed().toLower();
    normalized.remove(QLatin1Char(':'));
    return normalized;
}

std::optional<QByteArray> readFileBytes(const QString& path)
{
    if (path.trimmed().isEmpty()) {
        return std::nullopt;
    }
    QFile file(path);
    if (!file.exists()) {
        qCWarning(lcHealthClient) << "Plik TLS nie istnieje" << path;
        return std::nullopt;
    }
    if (!file.open(QIODevice::ReadOnly)) {
        qCWarning(lcHealthClient) << "Nie można odczytać pliku TLS" << path << file.errorString();
        return std::nullopt;
    }
    return file.readAll();
}

QString sha256Fingerprint(const QByteArray& pemData)
{
    const QList<QSslCertificate> certificates = QSslCertificate::fromData(pemData, QSsl::Pem);
    if (certificates.isEmpty()) {
        return {};
    }
    const QByteArray digest = certificates.first().digest(QCryptographicHash::Sha256);
    return QString::fromLatin1(digest.toHex()).toLower();
}

QStringList normalizeScopes(const QStringList& scopes)
{
    QStringList normalized;
    QSet<QString> seen;
    for (const QString& scope : scopes) {
        const QString trimmed = scope.trimmed();
        if (trimmed.isEmpty()) {
            continue;
        }
        if (seen.contains(trimmed)) {
            continue;
        }
        seen.insert(trimmed);
        normalized.append(trimmed);
    }
    return normalized;
}

} // namespace

HealthClient::HealthClient() = default;

HealthClient::~HealthClient() = default;

void HealthClient::setEndpoint(const QString& endpoint)
{
    const QString sanitized = endpoint.trimmed();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (sanitized == m_endpoint) {
        return;
    }
    m_endpoint = sanitized;
    resetChannelLocked();
}

void HealthClient::setTlsConfig(const GrpcTlsConfig& config)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    GrpcTlsConfig sanitized = config;
    sanitized.pinnedServerFingerprint = normalizeFingerprint(sanitized.pinnedServerFingerprint);
    if (sanitized == m_tlsConfig) {
        return;
    }
    m_tlsConfig = sanitized;
    resetChannelLocked();
}

void HealthClient::setAuthToken(const QString& token)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_authToken = token.trimmed();
}

void HealthClient::setRbacRole(const QString& role)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_rbacRole = role.trimmed();
}

void HealthClient::setRbacScopes(const QStringList& scopes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_rbacScopes = normalizeScopes(scopes);
}

QVector<QPair<QByteArray, QByteArray>> HealthClient::authMetadataForTesting() const
{
    QVector<QPair<QByteArray, QByteArray>> metadata;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_authToken.isEmpty()) {
        metadata.append({QByteArrayLiteral("authorization"),
                         QByteArrayLiteral("Bearer ") + m_authToken.toUtf8()});
    }
    const QStringList scopes = m_rbacScopes.isEmpty() ? QStringList{QStringLiteral("health.read")} : m_rbacScopes;
    for (const QString& scope : scopes) {
        metadata.append({QByteArrayLiteral("x-bot-scope"), scope.toUtf8()});
    }
    if (!m_rbacRole.isEmpty()) {
        metadata.append({QByteArrayLiteral("x-bot-role"), m_rbacRole.toUtf8()});
    }
    return metadata;
}

bool HealthClient::hasChannelForTesting() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return static_cast<bool>(m_channel);
}

bool HealthClient::hasStubForTesting() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return static_cast<bool>(m_stub);
}

HealthClient::HealthCheckResult HealthClient::check()
{
    ensureStub();
    std::unique_ptr<grpc::ClientContext> context;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_stub) {
            return {false, {}, {}, {}, QStringLiteral("Brak połączenia z HealthService")};
        }
        context = buildContext();
    }

    google::protobuf::Empty request;
    botcore::trading::v1::HealthCheckResponse response;
    const grpc::Status status = m_stub->Check(context.get(), request, &response);
    if (!status.ok()) {
        return {false, {}, {}, {}, QString::fromStdString(status.error_message())};
    }

    HealthCheckResult result;
    result.ok = true;
    result.version = QString::fromStdString(response.version());
    result.gitCommit = QString::fromStdString(response.git_commit());
    result.startedAtUtc = convertTimestamp(response);
    return result;
}

HealthClient::PreflightResult HealthClient::runPreflightChecklist() const
{
    PreflightResult result;

    QString endpoint;
    GrpcTlsConfig tls;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        endpoint = m_endpoint;
        tls = m_tlsConfig;
    }

    if (endpoint.trimmed().isEmpty()) {
        result.errors.append(QStringLiteral("Endpoint HealthService nie może być pusty."));
    }

    if (tls.enabled) {
        const QString rootPath = tls.rootCertificatePath.trimmed();
        if (rootPath.isEmpty()) {
            result.errors.append(QStringLiteral("Włączono TLS HealthService, ale nie wskazano pliku root CA."));
        } else if (!QFileInfo::exists(rootPath)) {
            result.errors.append(QStringLiteral("Plik root CA HealthService nie istnieje: %1").arg(rootPath));
        }

        const bool certProvided = !tls.clientCertificatePath.trimmed().isEmpty();
        const bool keyProvided = !tls.clientKeyPath.trimmed().isEmpty();
        if (tls.requireClientAuth && certProvided != keyProvided) {
            result.errors.append(QStringLiteral("Konfiguracja mTLS HealthService wymaga certyfikatu i klucza klienta."));
        }

        if (!tls.pinnedServerFingerprint.trimmed().isEmpty() && rootPath.isEmpty()) {
            result.warnings.append(QStringLiteral(
                "Pinning SHA-256 dla HealthService nie zostanie zweryfikowany bez pliku root CA."));
        }
    } else if (!tls.pinnedServerFingerprint.trimmed().isEmpty()) {
        result.warnings.append(QStringLiteral(
            "Podano fingerprint HealthService, ale TLS jest wyłączony – pinning zostanie pominięty."));
    }

    result.ok = result.errors.isEmpty();
    return result;
}

void HealthClient::ensureStub()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_endpoint.trimmed().isEmpty()) {
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
                if (const auto root = readFileBytes(m_tlsConfig.rootCertificatePath)) {
                    rootPem = *root;
                    options.pem_root_certs = std::string(rootPem.constData(), static_cast<std::size_t>(rootPem.size()));
                }
            }
            if (!m_tlsConfig.clientCertificatePath.trimmed().isEmpty()
                && !m_tlsConfig.clientKeyPath.trimmed().isEmpty()) {
                const auto cert = readFileBytes(m_tlsConfig.clientCertificatePath);
                const auto key = readFileBytes(m_tlsConfig.clientKeyPath);
                if (cert && key) {
                    grpc::SslCredentialsOptions::PemKeyCertPair pair;
                    pair.cert_chain = std::string(cert->constData(), static_cast<std::size_t>(cert->size()));
                    pair.private_key = std::string(key->constData(), static_cast<std::size_t>(key->size()));
                    options.pem_key_cert_pairs.push_back(std::move(pair));
                }
            }
            credentials = grpc::SslCredentials(options);

            if (!m_tlsConfig.targetNameOverride.trimmed().isEmpty()) {
                args.SetString(GRPC_SSL_TARGET_NAME_OVERRIDE_ARG, m_tlsConfig.targetNameOverride.toStdString());
            }
            if (!m_tlsConfig.serverNameOverride.trimmed().isEmpty()) {
                args.SetString(GRPC_ARG_OVERRIDE_DEFAULT_AUTHORITY, m_tlsConfig.serverNameOverride.toStdString());
            }

            if (!m_tlsConfig.pinnedServerFingerprint.isEmpty() && !rootPem.isEmpty()) {
                const QString fingerprint = sha256Fingerprint(rootPem);
                if (!fingerprint.isEmpty() && fingerprint != m_tlsConfig.pinnedServerFingerprint) {
                    qCWarning(lcHealthClient)
                        << "Fingerprint TLS HealthService nie zgadza się z konfiguracją."
                        << "Oczekiwano" << m_tlsConfig.pinnedServerFingerprint << "otrzymano" << fingerprint;
                    fingerprintValid = false;
                }
            }

            if (!fingerprintValid) {
                resetChannelLocked();
                return;
            }

            m_channel = grpc::CreateCustomChannel(m_endpoint.toStdString(), credentials, args);
        } else {
            credentials = grpc::InsecureChannelCredentials();
            m_channel = grpc::CreateChannel(m_endpoint.toStdString(), credentials);
        }
    }

    if (!m_stub && m_channel) {
        m_stub = botcore::trading::v1::HealthService::NewStub(m_channel);
    }
}

void HealthClient::resetChannelLocked()
{
    m_stub.reset();
    m_channel.reset();
}

std::unique_ptr<grpc::ClientContext> HealthClient::buildContext() const
{
    auto context = std::make_unique<grpc::ClientContext>();
    if (!m_authToken.trimmed().isEmpty()) {
        context->AddMetadata("authorization", std::string("Bearer ") + m_authToken.trimmed().toStdString());
    }
    const QStringList scopes = m_rbacScopes.isEmpty() ? QStringList{QStringLiteral("health.read")} : m_rbacScopes;
    for (const QString& scope : scopes) {
        context->AddMetadata("x-bot-scope", scope.toStdString());
    }
    if (!m_rbacRole.trimmed().isEmpty()) {
        context->AddMetadata("x-bot-role", m_rbacRole.trimmed().toStdString());
    }
    return context;
}

QDateTime HealthClient::convertTimestamp(const botcore::trading::v1::HealthCheckResponse& response)
{
    if (!response.has_started_at()) {
        return {};
    }
    const auto& ts = response.started_at();
    const qint64 seconds = static_cast<qint64>(ts.seconds());
    const qint64 nanos = static_cast<qint64>(ts.nanos());
    qint64 msecs = seconds * 1000;
    msecs += nanos / 1000000;
    return QDateTime::fromMSecsSinceEpoch(msecs, Qt::UTC);
}

