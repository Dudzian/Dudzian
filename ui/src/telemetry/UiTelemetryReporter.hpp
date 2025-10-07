#pragma once

#include <QObject>
#include <QJsonObject>
#include <QString>

#include <memory>
#include <mutex>
#include <optional>

#include "utils/PerformanceGuard.hpp"

namespace botcore::trading::v1 {
class MetricsService;
class MetricsSnapshot;
}

namespace grpc {
class Channel;
}

class UiTelemetryReporter : public QObject {
    Q_OBJECT
public:
    explicit UiTelemetryReporter(QObject* parent = nullptr);
    ~UiTelemetryReporter() override;

    void setEnabled(bool enabled);
    void setEndpoint(const QString& endpoint);
    void setNotesTag(const QString& tag);
    void setWindowCount(int count);
    void setAuthToken(const QString& token);
    bool isEnabled() const { return m_enabled && !m_endpoint.isEmpty(); }

    void reportReduceMotion(const PerformanceGuard& guard,
                             bool active,
                             double fps,
                             int overlayActive,
                             int overlayAllowed);
    void reportOverlayBudget(const PerformanceGuard& guard,
                             int overlayActive,
                             int overlayAllowed,
                             bool reduceMotionActive);

private:
    void pushSnapshot(const QJsonObject& notes, std::optional<double> fpsValue);
    botcore::trading::v1::MetricsService::Stub* ensureStub();

    bool m_enabled = false;
    QString m_endpoint;
    QString m_notesTag;
    QString m_authToken;
    int m_windowCount = 1;

    std::mutex m_mutex;
    std::shared_ptr<grpc::Channel> m_channel;
    std::unique_ptr<botcore::trading::v1::MetricsService::Stub> m_stub;
};

