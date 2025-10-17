#pragma once

#include <QRect>
#include <QString>

#include "telemetry/TelemetryTlsConfig.hpp"
#include "utils/PerformanceGuard.hpp"

class TelemetryReporter {
public:
    virtual ~TelemetryReporter() = default;

    struct ScreenInfo {
        QString name;
        QString manufacturer;
        QString model;
        QString serialNumber;
        QRect   geometry;
        QRect   availableGeometry;
        double  refreshRateHz = 0.0;
        double  devicePixelRatio = 1.0;
        double  logicalDpiX = 0.0;
        double  logicalDpiY = 0.0;
        int     index = -1;
    };

    virtual void setEnabled(bool enabled) = 0;
    virtual void setEndpoint(const QString& endpoint) = 0;
    virtual void setNotesTag(const QString& tag) = 0;
    virtual void setWindowCount(int count) = 0;
    virtual void setTlsConfig(const TelemetryTlsConfig& config) = 0;
    virtual void setAuthToken(const QString& token) = 0;
    virtual void setRbacRole(const QString& role) = 0;
    virtual void setScreenInfo(const ScreenInfo& info) = 0;
    virtual void clearScreenInfo() = 0;
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
    virtual void reportJankEvent(const PerformanceGuard& guard,
                                 double frameTimeMs,
                                 double thresholdMs,
                                 bool reduceMotionActive,
                                 int overlayActive,
                                 int overlayAllowed) = 0;
};

