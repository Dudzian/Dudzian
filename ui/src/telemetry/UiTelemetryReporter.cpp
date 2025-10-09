#include "UiTelemetryReporter.hpp"

#include <QCryptographicHash>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QtGlobal>

#include <chrono>
#include <optional>
#include <string>

#include <grpcpp/channel.h>
#include <grpcpp/channel_arguments.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "trading.grpc.pb.h"

Q_LOGGING_CATEGORY(lcTelemetry, "bot.shell.telemetry")

namespace {
QString buildNotesJson(const QJsonObject& base, const QString& tag, int windowCount) {
    QJsonObject enriched = base;
    if (!tag.isEmpty()) {
        enriched.insert(QStringLiteral("tag"), tag);
    }
    enriched.insert(QStringLiteral("window_count"), windowCount);
    return QString::fromUtf8(QJsonDocument(enriched).toJson(QJsonDocument::Compact));
}

void stampNow(botcore::trading::v1::MetricsSnapshot& snapshot) {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto secondsPart = duration_cast<seconds>(now.time_since_epoch());
    const auto nanosPart = duration_cast<nanoseconds>(now.time_since_epoch())
                         - duration_cast<nanoseconds>(secondsPart);
    auto* ts = snapshot.mutable_generated_at();
    ts->set_seconds(secondsPart.count());
    ts->set_nanos(static_cast<int32_t>(nanosPart.count()));
}
} // namespace

UiTelemetryReporter::UiTelemetryReporter(QObject* parent)
    : QObject(parent) {}

UiTelemetryReporter::~UiTelemetryReporter() = default;

void UiTelemetryReporter::setEnabled(bool enabled) {
    if (m_enabled == enabled) {
        return;
    }
    m_enabled = enabled;
}

void UiTelemetryReporter::setEndpoint(const QString& endpoint) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_endpoint == endpoint) {
        return;
    }
    m_endpoint = endpoint;
    m_channel.reset();
    m_stub.reset();
}

void UiTelemetryReporter::setNotesTag(const QString& tag) {
    m_notesTag = tag;
}

void UiTelemetryReporter::setWindowCount(int count) {
    m_windowCount = qMax(1, count);
}

void UiTelemetryReporter::setTlsConfig(const TelemetryTlsConfig& config) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_tlsConfig = config;
    if (!m_tlsConfig.enabled) {
        m_tlsConfig.rootCertificatePath.clear();
        m_tlsConfig.clientCertificatePath.clear();
        m_tlsConfig.clientKeyPath.clear();
        m_tlsConfig.serverNameOverride.clear();
        m_tlsConfig.pinnedServerSha256.clear();
    }
    m_channel.reset();
    m_stub.reset();
}

void UiTelemetryReporter::setAuthToken(const QString& token) {
    m_authToken = token;
}

void UiTelemetryReporter::reportReduceMotion(const PerformanceGuard& guard,
                                             bool active,
                                             double fps,
                                             int overlayActive,
                                             int overlayAllowed) {
    QJsonObject payload{
        {QStringLiteral("event"), QStringLiteral("reduce_motion")},
        {QStringLiteral("active"), active},
        {QStringLiteral("fps_target"), guard.fpsTarget},
        {QStringLiteral("overlay_active"), overlayActive},
        {QStringLiteral("overlay_allowed"), overlayAllowed},
        {QStringLiteral("jank_budget_ms"), guard.jankThresholdMs}
    };
    if (guard.disableSecondaryWhenFpsBelow > 0) {
        payload.insert(QStringLiteral("disable_secondary_fps"), guard.disableSecondaryWhenFpsBelow);
    }
    pushSnapshot(payload, fps > 0.0 ? std::optional<double>(fps) : std::nullopt);
}

void UiTelemetryReporter::reportOverlayBudget(const PerformanceGuard& guard,
                                              int overlayActive,
                                              int overlayAllowed,
                                              bool reduceMotionActive) {
    QJsonObject payload{
        {QStringLiteral("event"), QStringLiteral("overlay_budget")},
        {QStringLiteral("active_overlays"), overlayActive},
        {QStringLiteral("allowed_overlays"), overlayAllowed},
        {QStringLiteral("reduce_motion"), reduceMotionActive},
        {QStringLiteral("fps_target"), guard.fpsTarget}
    };
    if (guard.disableSecondaryWhenFpsBelow > 0) {
        payload.insert(QStringLiteral("disable_secondary_fps"), guard.disableSecondaryWhenFpsBelow);
    }
    pushSnapshot(payload, std::nullopt);
}

void UiTelemetryReporter::pushSnapshot(const QJsonObject& notes, std::optional<double> fpsValue) {
    if (!m_enabled || m_endpoint.isEmpty()) {
        return;
    }

    auto* stub = ensureStub();
    if (!stub) {
        qCWarning(lcTelemetry) << "Metrics stub unavailable for" << m_endpoint;
        return;
    }

    botcore::trading::v1::MetricsSnapshot snapshot;
    stampNow(snapshot);
    if (fpsValue.has_value()) {
        snapshot.set_fps(fpsValue.value());
    }
    snapshot.set_notes(buildNotesJson(notes, m_notesTag, m_windowCount).toStdString());

    grpc::ClientContext context;
    if (!m_authToken.isEmpty()) {
        const std::string token = m_authToken.toStdString();
        context.AddMetadata("authorization", std::string("Bearer ") + token);
    }

    botcore::trading::v1::MetricsAck ack;
    const auto status = stub->PushMetrics(&context, snapshot, &ack);
    if (!status.ok()) {
        qCWarning(lcTelemetry) << "PushMetrics failed"
                               << QString::fromStdString(status.error_message());
    }
}

namespace {

std::optional<std::string> readFileUtf8(const QString& path) {
    if (path.isEmpty()) {
        return std::nullopt;
    }
    QFile file(path);
    if (!file.exists()) {
        return std::nullopt;
    }
    if (!file.open(QIODevice::ReadOnly)) {
        qCWarning(lcTelemetry) << "Nie można otworzyć pliku TLS" << path << file.errorString();
        return std::nullopt;
    }
    const QByteArray data = file.readAll();
    return std::string(data.constData(), static_cast<std::size_t>(data.size()));
}

void verifyPinnedFingerprint(const TelemetryTlsConfig& config, const QByteArray& certificate) {
    if (config.pinnedServerSha256.isEmpty() || certificate.isEmpty()) {
        return;
    }
    const QByteArray digest = QCryptographicHash::hash(certificate, QCryptographicHash::Sha256).toHex();
    const QString hex = QString::fromUtf8(digest);
    if (hex.compare(config.pinnedServerSha256, Qt::CaseInsensitive) != 0) {
        qCWarning(lcTelemetry) << "Niepoprawny odcisk SHA-256 certyfikatu serwera MetricsService";
    }
}

} // namespace

botcore::trading::v1::MetricsService::Stub* UiTelemetryReporter::ensureStub() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_endpoint.isEmpty()) {
        return nullptr;
    }
    if (!m_channel) {
        std::shared_ptr<grpc::ChannelCredentials> credentials;
        grpc::ChannelArguments args;
        if (m_tlsConfig.enabled) {
            grpc::SslCredentialsOptions options;
            QByteArray rootPem;
            if (!m_tlsConfig.rootCertificatePath.isEmpty()) {
                QFile rootFile(m_tlsConfig.rootCertificatePath);
                if (rootFile.open(QIODevice::ReadOnly)) {
                    rootPem = rootFile.readAll();
                    options.pem_root_certs = std::string(rootPem.constData(), static_cast<std::size_t>(rootPem.size()));
                } else {
                    qCWarning(lcTelemetry) << "Nie udało się odczytać root CA" << m_tlsConfig.rootCertificatePath
                                           << rootFile.errorString();
                }
            }
            if (!m_tlsConfig.clientCertificatePath.isEmpty() && !m_tlsConfig.clientKeyPath.isEmpty()) {
                auto cert = readFileUtf8(m_tlsConfig.clientCertificatePath);
                auto key  = readFileUtf8(m_tlsConfig.clientKeyPath);
                if (cert && key) {
                    options.pem_key_cert_pairs.push_back({*key, *cert});
                } else {
                    qCWarning(lcTelemetry) << "Nie można odczytać certyfikatu lub klucza klienta TLS";
                }
            }
            credentials = grpc::SslCredentials(options);
            if (!m_tlsConfig.serverNameOverride.isEmpty()) {
                args.SetString(GRPC_SSL_TARGET_NAME_OVERRIDE_ARG,
                               m_tlsConfig.serverNameOverride.toStdString());
            }
            if (!rootPem.isEmpty()) {
                verifyPinnedFingerprint(m_tlsConfig, rootPem);
            }
        } else {
            credentials = grpc::InsecureChannelCredentials();
        }

        if (!credentials) {
            qCWarning(lcTelemetry) << "Brak kredencji TLS dla MetricsService";
            return nullptr;
        }
        if (m_tlsConfig.enabled) {
            m_channel = grpc::CreateCustomChannel(m_endpoint.toStdString(), credentials, args);
        } else {
            m_channel = grpc::CreateChannel(m_endpoint.toStdString(), credentials);
        }
    }
    if (!m_stub && m_channel) {
        m_stub = botcore::trading::v1::MetricsService::NewStub(m_channel);
    }
    return m_stub.get();
}
