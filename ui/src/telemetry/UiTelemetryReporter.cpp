#include "UiTelemetryReporter.hpp"

#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QRect>
#include <QtGlobal>

#include <chrono>
#include <optional>
#include <string>

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
    : QObject(parent)
    , m_client(std::make_shared<MetricsClient>()) {}

UiTelemetryReporter::~UiTelemetryReporter() = default;

void UiTelemetryReporter::setEnabled(bool enabled) {
    if (m_enabled == enabled) {
        return;
    }
    m_enabled = enabled;
    if (!m_enabled) {
        resetRetryBuffer();
    }
}

void UiTelemetryReporter::setEndpoint(const QString& endpoint) {
    if (m_endpoint == endpoint) {
        return;
    }
    m_endpoint = endpoint;
    if (m_client) {
        m_client->setEndpoint(endpoint);
    }
}

void UiTelemetryReporter::setNotesTag(const QString& tag) {
    m_notesTag = tag;
}

void UiTelemetryReporter::setWindowCount(int count) {
    m_windowCount = qMax(1, count);
}

void UiTelemetryReporter::setTlsConfig(const TelemetryTlsConfig& config) {
    m_tlsConfig = config;
    if (!m_tlsConfig.enabled) {
        m_tlsConfig.rootCertificatePath.clear();
        m_tlsConfig.clientCertificatePath.clear();
        m_tlsConfig.clientKeyPath.clear();
        m_tlsConfig.serverNameOverride.clear();
        m_tlsConfig.pinnedServerSha256.clear();
    }
    if (m_client) {
        m_client->setTlsConfig(m_tlsConfig);
    }
}

void UiTelemetryReporter::setAuthToken(const QString& token) {
    m_authToken = token;
    if (m_client) {
        m_client->setAuthToken(token);
    }
}

void UiTelemetryReporter::setRbacRole(const QString& role) {
    m_rbacRole = role.trimmed();
    if (m_client) {
        m_client->setRbacRole(m_rbacRole);
    }
}

void UiTelemetryReporter::setScreenInfo(const ScreenInfo& info) {
    m_screenInfo = info;
}

void UiTelemetryReporter::clearScreenInfo() {
    m_screenInfo.reset();
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

void UiTelemetryReporter::reportJankEvent(const PerformanceGuard& guard,
                                          double frameTimeMs,
                                          double thresholdMs,
                                          bool reduceMotionActive,
                                          int overlayActive,
                                          int overlayAllowed) {
    QJsonObject payload{
        {QStringLiteral("event"), QStringLiteral("jank_spike")},
        {QStringLiteral("frame_ms"), frameTimeMs},
        {QStringLiteral("threshold_ms"), thresholdMs},
        {QStringLiteral("reduce_motion"), reduceMotionActive},
        {QStringLiteral("overlay_active"), overlayActive},
        {QStringLiteral("overlay_allowed"), overlayAllowed},
        {QStringLiteral("fps_target"), guard.fpsTarget},
    };
    if (guard.disableSecondaryWhenFpsBelow > 0) {
        payload.insert(QStringLiteral("disable_secondary_fps"), guard.disableSecondaryWhenFpsBelow);
    }
    if (guard.jankThresholdMs > 0.0) {
        payload.insert(QStringLiteral("configured_jank_threshold_ms"), guard.jankThresholdMs);
    }
    if (thresholdMs > 0.0 && frameTimeMs > thresholdMs) {
        const double overBudget = frameTimeMs - thresholdMs;
        payload.insert(QStringLiteral("over_budget_ms"), overBudget);
        payload.insert(QStringLiteral("ratio"), frameTimeMs / thresholdMs);
    }
    const double fpsEstimate = frameTimeMs > 0.0 ? 1000.0 / frameTimeMs : 0.0;
    pushSnapshot(payload, fpsEstimate > 0.0 ? std::optional<double>(fpsEstimate) : std::nullopt);
}

void UiTelemetryReporter::setMetricsClientForTesting(const std::shared_ptr<MetricsClientInterface>& client) {
    if (!client) {
        return;
    }
    m_client = client;
    m_client->setEndpoint(m_endpoint);
    m_client->setTlsConfig(m_tlsConfig);
    m_client->setAuthToken(m_authToken);
    m_client->setRbacRole(m_rbacRole);
}

void UiTelemetryReporter::setRetryBufferLimitForTesting(int limit) {
    m_retryBufferLimit = qMax(0, limit);
    while (static_cast<int>(m_retryBuffer.size()) > m_retryBufferLimit) {
        m_retryBuffer.pop_front();
    }
    publishRetryBufferSizeIfNeeded();
}

void UiTelemetryReporter::pushSnapshot(const QJsonObject& notes, std::optional<double> fpsValue) {
    if (!m_enabled || m_endpoint.isEmpty() || !m_client) {
        return;
    }

    const int backlogBeforeFlush = pendingRetryCount();
    flushRetryBuffer();

    QJsonObject enrichedNotes = notes;
    enrichedNotes.insert(QStringLiteral("retry_backlog_before_send"), backlogBeforeFlush);
    enrichedNotes.insert(QStringLiteral("retry_backlog_after_flush"), pendingRetryCount());

    if (m_screenInfo.has_value()) {
        enrichedNotes.insert(QStringLiteral("screen"), buildScreenJson());
    }

    botcore::trading::v1::MetricsSnapshot snapshot;
    stampNow(snapshot);
    if (fpsValue.has_value()) {
        snapshot.set_fps(fpsValue.value());
    }
    snapshot.set_notes(buildNotesJson(enrichedNotes, m_notesTag, m_windowCount).toStdString());

    QString error;
    if (!m_client->pushSnapshot(snapshot, &error)) {
        if (!error.isEmpty()) {
            qCWarning(lcTelemetry) << "PushMetrics failed" << error;
        } else {
            qCWarning(lcTelemetry) << "PushMetrics failed (unknown error)";
        }
        if (m_retryBufferLimit > 0) {
            if (static_cast<int>(m_retryBuffer.size()) >= m_retryBufferLimit) {
                qCWarning(lcTelemetry) << "Bufor retry telemetry pełny, najstarsza próbka zostanie odrzucona";
                m_retryBuffer.pop_front();
            }
            m_retryBuffer.push_back(snapshot);
            publishRetryBufferSizeIfNeeded();
        }
    }
}

void UiTelemetryReporter::flushRetryBuffer() {
    if (!m_client || m_retryBuffer.empty()) {
        return;
    }
    bool changed = false;
    for (int attempt = 0; attempt < m_retryBufferLimit && !m_retryBuffer.empty(); ++attempt) {
        botcore::trading::v1::MetricsSnapshot& pending = m_retryBuffer.front();
        QString error;
        if (!m_client->pushSnapshot(pending, &error)) {
            if (!error.isEmpty()) {
                qCWarning(lcTelemetry) << "Ponowna wysyłka telemetrii nie powiodła się" << error;
            } else {
                qCWarning(lcTelemetry) << "Ponowna wysyłka telemetrii nie powiodła się (unknown error)";
            }
            break;
        }
        m_retryBuffer.pop_front();
        changed = true;
    }
    if (changed) {
        publishRetryBufferSizeIfNeeded();
    }
}

QJsonObject UiTelemetryReporter::buildScreenJson() const {
    QJsonObject screen;
    if (!m_screenInfo.has_value()) {
        return screen;
    }
    const ScreenInfo info = m_screenInfo.value();
    screen.insert(QStringLiteral("name"), info.name);
    if (!info.manufacturer.isEmpty()) {
        screen.insert(QStringLiteral("manufacturer"), info.manufacturer);
    }
    if (!info.model.isEmpty()) {
        screen.insert(QStringLiteral("model"), info.model);
    }
    if (!info.serialNumber.isEmpty()) {
        screen.insert(QStringLiteral("serial"), info.serialNumber);
    }
    screen.insert(QStringLiteral("index"), info.index);
    screen.insert(QStringLiteral("refresh_hz"), info.refreshRateHz);
    screen.insert(QStringLiteral("device_pixel_ratio"), info.devicePixelRatio);
    screen.insert(QStringLiteral("logical_dpi_x"), info.logicalDpiX);
    screen.insert(QStringLiteral("logical_dpi_y"), info.logicalDpiY);

    const auto rectToJson = [](const QRect& rect) {
        QJsonObject obj;
        obj.insert(QStringLiteral("x"), rect.x());
        obj.insert(QStringLiteral("y"), rect.y());
        obj.insert(QStringLiteral("width"), rect.width());
        obj.insert(QStringLiteral("height"), rect.height());
        return obj;
    };
    screen.insert(QStringLiteral("geometry_px"), rectToJson(info.geometry));
    screen.insert(QStringLiteral("available_geometry_px"), rectToJson(info.availableGeometry));
    return screen;
}

int UiTelemetryReporter::pendingRetryCount() const {
    return static_cast<int>(m_retryBuffer.size());
}

void UiTelemetryReporter::resetRetryBuffer() {
    if (m_retryBuffer.empty()) {
        return;
    }
    m_retryBuffer.clear();
    publishRetryBufferSizeIfNeeded();
}

void UiTelemetryReporter::publishRetryBufferSizeIfNeeded() {
    const int current = pendingRetryCount();
    if (current == m_lastPublishedRetryCount) {
        return;
    }
    m_lastPublishedRetryCount = current;
    Q_EMIT pendingRetryCountChanged(current);
}

