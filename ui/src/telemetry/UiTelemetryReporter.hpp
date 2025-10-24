#pragma once

#include <QObject>
#include <QJsonObject>
#include <QString>

#include <deque>
#include <memory>
#include <optional>

#include "telemetry/TelemetryReporter.hpp"
#include "telemetry/TelemetryTlsConfig.hpp"
#include "utils/PerformanceGuard.hpp"

#include "grpc/MetricsClient.hpp"

class UiTelemetryReporter : public QObject, public TelemetryReporter {
    Q_OBJECT
    Q_PROPERTY(int pendingRetryCount READ pendingRetryCount NOTIFY pendingRetryCountChanged)
public:
    explicit UiTelemetryReporter(QObject* parent = nullptr);
    ~UiTelemetryReporter() override;

    // TelemetryReporter API
    void setEnabled(bool enabled) override;
    void setEndpoint(const QString& endpoint) override;
    void setNotesTag(const QString& tag) override;
    void setWindowCount(int count) override;
    void setTlsConfig(const TelemetryTlsConfig& config) override;
    void setAuthToken(const QString& token) override;
    void setRbacRole(const QString& role) override;
    void setScreenInfo(const ScreenInfo& info) override;
    void clearScreenInfo() override;
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

    void reportJankEvent(const PerformanceGuard& guard,
                         double frameTimeMs,
                         double thresholdMs,
                         bool reduceMotionActive,
                         int overlayActive,
                         int overlayAllowed) override;

    void setMetricsClientForTesting(const std::shared_ptr<MetricsClientInterface>& client);
    void setRetryBufferLimitForTesting(int limit);

    int pendingRetryCount() const;

private:
    friend class PerformanceTelemetryController;
    void pushSnapshot(const QJsonObject& notes, std::optional<double> fpsValue);
    void flushRetryBuffer();
    void resetRetryBuffer();
    void publishRetryBufferSizeIfNeeded();
    QJsonObject buildScreenJson() const;

    // Konfiguracja / stan
    bool        m_enabled = false;
    QString     m_endpoint;
    QString     m_notesTag;
    QString     m_authToken;
    QString     m_rbacRole;
    int         m_windowCount = 1;
    std::optional<ScreenInfo> m_screenInfo;

    // gRPC client wrapper i retry buffer
    std::shared_ptr<MetricsClientInterface> m_client;
    std::deque<botcore::trading::v1::MetricsSnapshot> m_retryBuffer;
    int m_retryBufferLimit = 16;
    TelemetryTlsConfig m_tlsConfig;
    int m_lastPublishedRetryCount = 0;

Q_SIGNALS:
    void pendingRetryCountChanged(int pending);
};
