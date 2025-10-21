#include "MetricsClient.hpp"

#include <QCryptographicHash>
#include <QFile>
#include <QFileInfo>
#include <QLoggingCategory>
#include <QIODevice>
#include <QStringList>

#include <grpcpp/channel.h>
#include <grpcpp/channel_arguments.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/security/credentials.h>

#include "trading.grpc.pb.h"

Q_LOGGING_CATEGORY(lcMetricsClient, "bot.shell.telemetry.grpc")

namespace {

QString normalizeFingerprint(QString value)
{
    QString normalized = value.trimmed().toLower();
    normalized.remove(QLatin1Char(':'));
    return normalized;
}

std::optional<std::string> readFileUtf8(const QString& path)
{
    if (path.trimmed().isEmpty()) {
        return std::nullopt;
    }
    QFile file(path);
    if (!file.exists()) {
        qCWarning(lcMetricsClient) << "Plik TLS nie istnieje" << path;
        return std::nullopt;
    }
    if (!file.open(QIODevice::ReadOnly)) {
        qCWarning(lcMetricsClient) << "Nie można otworzyć pliku TLS" << path << file.errorString();
        return std::nullopt;
    }
    const QByteArray data = file.readAll();
    return std::string(data.constData(), static_cast<std::size_t>(data.size()));
}

void verifyPinnedFingerprint(const TelemetryTlsConfig& config, const QByteArray& certificate)
{
    if (config.pinnedServerSha256.isEmpty() || certificate.isEmpty()) {
        return;
    }
    const QByteArray digest = QCryptographicHash::hash(certificate, QCryptographicHash::Sha256).toHex();
    const QString hex = QString::fromUtf8(digest).toLower();
    if (hex != config.pinnedServerSha256) {
        qCWarning(lcMetricsClient) << "Niepoprawny odcisk SHA-256 certyfikatu serwera MetricsService";
    }
}

} // namespace

MetricsClient::MetricsClient() = default;

MetricsClient::~MetricsClient() = default;

void MetricsClient::setEndpoint(const QString& endpoint)
{
    const QString sanitized = endpoint.trimmed();
    if (sanitized == m_endpoint) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_endpoint = sanitized;
    m_channel.reset();
    m_stub.reset();
}

void MetricsClient::setTlsConfig(const TelemetryTlsConfig& config)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    TelemetryTlsConfig sanitized = config;
    sanitized.pinnedServerSha256 = normalizeFingerprint(sanitized.pinnedServerSha256);
    m_tlsConfig = sanitized;
    m_channel.reset();
    m_stub.reset();
}

void MetricsClient::setAuthToken(const QString& token)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_authToken = token.trimmed();
}

void MetricsClient::setRbacRole(const QString& role)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_rbacRole = role.trimmed();
}

bool MetricsClient::pushSnapshot(const botcore::trading::v1::MetricsSnapshot& snapshot, QString* errorMessage)
{
    ensureChannel();
    std::unique_ptr<grpc::ClientContext> context;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_stub) {
            if (errorMessage) {
                *errorMessage = QStringLiteral("Brak połączenia z MetricsService");
            }
            return false;
        }
        context = buildContext();
    }

    botcore::trading::v1::MetricsAck ack;
    const grpc::Status status = m_stub->PushMetrics(context.get(), snapshot, &ack);
    if (!status.ok()) {
        if (errorMessage) {
            *errorMessage = QString::fromStdString(status.error_message());
        }
        qCWarning(lcMetricsClient) << "PushMetrics failed" << QString::fromStdString(status.error_message());
        return false;
    }
    return true;
}

QVector<QPair<QByteArray, QByteArray>> MetricsClient::authMetadataForTesting() const
{
    QVector<QPair<QByteArray, QByteArray>> metadata;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_authToken.isEmpty()) {
        metadata.append({QByteArrayLiteral("authorization"),
                         QByteArrayLiteral("Bearer ") + m_authToken.toUtf8()});
    }
    metadata.append({QByteArrayLiteral("x-bot-scope"), QByteArrayLiteral("metrics.write")});
    if (!m_rbacRole.isEmpty()) {
        metadata.append({QByteArrayLiteral("x-bot-role"), m_rbacRole.toUtf8()});
    }
    return metadata;
}

MetricsPreflightResult MetricsClient::runPreflightChecklist() const
{
    MetricsPreflightResult result;
    QString endpoint;
    TelemetryTlsConfig tls;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        endpoint = m_endpoint;
        tls = m_tlsConfig;
    }

    if (endpoint.trimmed().isEmpty()) {
        result.errors.append(QStringLiteral("Endpoint MetricsService nie może być pusty."));
    }

    if (tls.enabled) {
        const QString rootPath = tls.rootCertificatePath.trimmed();
        if (rootPath.isEmpty()) {
            result.errors.append(QStringLiteral("Włączono TLS telemetrii, ale nie wskazano pliku root CA."));
        } else {
            QFileInfo rootInfo(rootPath);
            if (!rootInfo.exists()) {
                result.errors.append(QStringLiteral("Plik root CA MetricsService nie istnieje: %1").arg(rootPath));
            } else {
                QFile rootFile(rootPath);
                if (!rootFile.open(QIODevice::ReadOnly)) {
                    result.errors.append(QStringLiteral("Nie można odczytać root CA MetricsService (%1): %2")
                                             .arg(rootPath, rootFile.errorString()));
                } else {
                    const QByteArray pem = rootFile.readAll();
                    if (pem.isEmpty()) {
                        result.errors.append(QStringLiteral("Plik root CA MetricsService jest pusty."));
                    } else if (!tls.pinnedServerSha256.isEmpty()) {
                        const QString digest = QString::fromUtf8(
                            QCryptographicHash::hash(pem, QCryptographicHash::Sha256).toHex());
                        if (digest != tls.pinnedServerSha256) {
                            result.errors.append(QStringLiteral(
                                "Fingerprint root CA nie pasuje do metrics-server-sha256."));
                        }
                    }
                }
            }
        }

        const QString certPath = tls.clientCertificatePath.trimmed();
        const QString keyPath = tls.clientKeyPath.trimmed();
        const bool certProvided = !certPath.isEmpty();
        const bool keyProvided = !keyPath.isEmpty();
        if (certProvided != keyProvided) {
            result.errors.append(QStringLiteral(
                "Konfiguracja mTLS MetricsService wymaga zarówno certyfikatu, jak i klucza klienta."));
        }
        if (certProvided) {
            if (!QFileInfo::exists(certPath)) {
                result.errors.append(QStringLiteral("Certyfikat klienta MetricsService nie istnieje: %1")
                                         .arg(certPath));
            }
            if (!QFileInfo::exists(keyPath)) {
                result.errors.append(QStringLiteral("Klucz klienta MetricsService nie istnieje: %1")
                                         .arg(keyPath));
            }
        }
    } else if (!tls.pinnedServerSha256.isEmpty()) {
        result.warnings.append(QStringLiteral(
            "Ustawiono metrics-server-sha256, ale TLS telemetrii jest wyłączony."));
    }

    result.ok = result.errors.isEmpty();
    return result;
}

void MetricsClient::ensureChannel()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_endpoint.trimmed().isEmpty()) {
        return;
    }
    if (!m_channel) {
        std::shared_ptr<grpc::ChannelCredentials> credentials;
        grpc::ChannelArguments args;
        QByteArray rootPem;
        if (m_tlsConfig.enabled) {
            grpc::SslCredentialsOptions options;
            if (!m_tlsConfig.rootCertificatePath.isEmpty()) {
                QFile rootFile(m_tlsConfig.rootCertificatePath);
                if (rootFile.open(QIODevice::ReadOnly)) {
                    rootPem = rootFile.readAll();
                    options.pem_root_certs = std::string(rootPem.constData(), static_cast<std::size_t>(rootPem.size()));
                } else {
                    qCWarning(lcMetricsClient) << "Nie udało się odczytać root CA" << m_tlsConfig.rootCertificatePath
                                               << rootFile.errorString();
                }
            }
            if (!m_tlsConfig.clientCertificatePath.isEmpty() && !m_tlsConfig.clientKeyPath.isEmpty()) {
                auto cert = readFileUtf8(m_tlsConfig.clientCertificatePath);
                auto key = readFileUtf8(m_tlsConfig.clientKeyPath);
                if (cert && key) {
                    options.pem_key_cert_pairs.push_back({*key, *cert});
                }
            }
            credentials = grpc::SslCredentials(options);
            if (!m_tlsConfig.serverNameOverride.isEmpty()) {
                args.SetString(GRPC_SSL_TARGET_NAME_OVERRIDE_ARG, m_tlsConfig.serverNameOverride.toStdString());
            }
            if (!rootPem.isEmpty()) {
                verifyPinnedFingerprint(m_tlsConfig, rootPem);
            }
            m_channel = grpc::CreateCustomChannel(m_endpoint.toStdString(), credentials, args);
        } else {
            credentials = grpc::InsecureChannelCredentials();
            m_channel = grpc::CreateChannel(m_endpoint.toStdString(), credentials);
        }
    }
    if (!m_stub && m_channel) {
        m_stub = botcore::trading::v1::MetricsService::NewStub(m_channel);
    }
}

std::unique_ptr<grpc::ClientContext> MetricsClient::buildContext() const
{
    auto context = std::make_unique<grpc::ClientContext>();
    if (!m_authToken.trimmed().isEmpty()) {
        const std::string token = m_authToken.trimmed().toStdString();
        context->AddMetadata("authorization", std::string("Bearer ") + token);
    }
    context->AddMetadata("x-bot-scope", "metrics.write");
    if (!m_rbacRole.isEmpty()) {
        context->AddMetadata("x-bot-role", m_rbacRole.toStdString());
    }
    return context;
}

