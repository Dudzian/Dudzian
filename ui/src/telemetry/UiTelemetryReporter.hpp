#pragma once

#include <QObject>
#include <QJsonObject>
#include <QString>

#include <memory>
#include <mutex>
#include <optional>

#include "telemetry/TelemetryReporter.hpp"
#include "telemetry/TelemetryTlsConfig.hpp"
#include "utils/PerformanceGuard.hpp"

namespace botcore::trading::v1 {
class MetricsService;
class MetricsSnapshot;
}

namespace grpc {
class Channel;
}

class UiTelemetryReporter : public QObject, public TelemetryReporter {
    Q_OBJECT
public:
    explicit UiTelemetryReporter(QObject* parent = nullptr);
    ~UiTelemetryReporter() override;

    void setEnabled(bool enabled) override;
    void setEndpoint(const QString& endpoint) override;
    void setNotesTag(const QString& tag) override;
    void setWindowCount(int count) override;
    void setTlsConfig(const TelemetryTlsConfig& config) override;
    bool isEnabled() const override { return m_enabled && !m_endpoint.isEmpty(); }

    void reportReduceMotion(const PerformanceGuard& guard,
                             bool active,
                             double fps,
                             int overlayActive,
                             int overlayAllowed) override;
    void reportOverlayBudget(const PerformanceGuard& guard,
                             int overlayActive,
                             int overlayAllowed,
                             bool reduceMotionActive) override;

private:
    void pushSnapshot(const QJsonObject& notes, std::optional<double> fpsValue);
    botcore::trading::v1::MetricsService::Stub* ensureStub();

    bool m_enabled = false;
    QString m_endpoint;
    QString m_notesTag;
    int m_windowCount = 1;

    std::mutex m_mutex;
    std::shared_ptr<grpc::Channel> m_channel;
    std::unique_ptr<botcore::trading::v1::MetricsService::Stub> m_stub;
    TelemetryTlsConfig m_tlsConfig;
};

