#pragma once

#include <QString>

#include "telemetry/TelemetryTlsConfig.hpp"
#include "utils/PerformanceGuard.hpp"

class TelemetryReporter {
public:
    virtual ~TelemetryReporter() = default;

    virtual void setEnabled(bool enabled) = 0;
    virtual void setEndpoint(const QString& endpoint) = 0;
    virtual void setNotesTag(const QString& tag) = 0;
    virtual void setWindowCount(int count) = 0;
    virtual void setTlsConfig(const TelemetryTlsConfig& config) = 0;
    virtual bool isEnabled() const = 0;

    virtual void reportReduceMotion(const PerformanceGuard& guard,
                                    bool active,
                                    double fps,
                                    int overlayActive,
                                    int overlayAllowed) = 0;
    virtual void reportOverlayBudget(const PerformanceGuard& guard,
                                     int overlayActive,
                                     int overlayAllowed,
                                     bool reduceMotionActive) = 0;
};

