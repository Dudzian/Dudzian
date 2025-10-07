#include "UiTelemetryReporter.hpp"

#include <QJsonDocument>
#include <QLoggingCategory>
#include <QtGlobal>

#include <chrono>
#include <string>

#include <grpcpp/channel.h>
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
    const auto nanosPart = duration_cast<nanoseconds>(now.time_since_epoch()) - duration_cast<nanoseconds>(secondsPart);
    auto* ts = snapshot.mutable_generated_at();
    ts->set_seconds(secondsPart.count());
    ts->set_nanos(static_cast<int32_t>(nanosPart.count()));
}
} // namespace

UiTelemetryReporter::UiTelemetryReporter(QObject* parent)
    : QObject(parent) {
}

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

void UiTelemetryReporter::setAuthToken(const QString& token) {
    m_authToken = token;
}

void UiTelemetryReporter::reportReduceMotion(const PerformanceGuard& guard,
                                             bool active,
                                             double fps,
                                             int overlayActive,
                                             int overlayAllowed) {
    QJsonObject payload{{QStringLiteral("event"), QStringLiteral("reduce_motion")},
                        {QStringLiteral("active"), active},
                        {QStringLiteral("fps_target"), guard.fpsTarget},
                        {QStringLiteral("overlay_active"), overlayActive},
                        {QStringLiteral("overlay_allowed"), overlayAllowed},
                        {QStringLiteral("jank_budget_ms"), guard.jankThresholdMs}};
    if (guard.disableSecondaryWhenFpsBelow > 0) {
        payload.insert(QStringLiteral("disable_secondary_fps"), guard.disableSecondaryWhenFpsBelow);
    }
    pushSnapshot(payload, fps > 0.0 ? std::optional<double>(fps) : std::nullopt);
}

void UiTelemetryReporter::reportOverlayBudget(const PerformanceGuard& guard,
                                              int overlayActive,
                                              int overlayAllowed,
                                              bool reduceMotionActive) {
    QJsonObject payload{{QStringLiteral("event"), QStringLiteral("overlay_budget")},
                        {QStringLiteral("active_overlays"), overlayActive},
                        {QStringLiteral("allowed_overlays"), overlayAllowed},
                        {QStringLiteral("reduce_motion"), reduceMotionActive},
                        {QStringLiteral("fps_target"), guard.fpsTarget}};
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
    const auto status = stub->PushMetrics(&context, snapshot);
    if (!status.ok()) {
        qCWarning(lcTelemetry) << "PushMetrics failed" << QString::fromStdString(status.error_message());
    }
}

botcore::trading::v1::MetricsService::Stub* UiTelemetryReporter::ensureStub() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_endpoint.isEmpty()) {
        return nullptr;
    }
    if (!m_channel) {
        m_channel = grpc::CreateChannel(m_endpoint.toStdString(), grpc::InsecureChannelCredentials());
    }
    if (!m_stub && m_channel) {
        m_stub = botcore::trading::v1::MetricsService::NewStub(m_channel);
    }
    return m_stub.get();
}

