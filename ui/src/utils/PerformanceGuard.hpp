#pragma once

#include <QObject>
#include <QVariantMap>

struct PerformanceGuard {
    Q_GADGET
    Q_PROPERTY(int fpsTarget MEMBER fpsTarget)
    Q_PROPERTY(double reduceMotionAfterSeconds MEMBER reduceMotionAfterSeconds)
    Q_PROPERTY(double jankThresholdMs MEMBER jankThresholdMs)
    Q_PROPERTY(int maxOverlayCount MEMBER maxOverlayCount)

public:
    int fpsTarget = 60;
    double reduceMotionAfterSeconds = 1.0;
    double jankThresholdMs = 18.0;
    int maxOverlayCount = 3;
};

Q_DECLARE_METATYPE(PerformanceGuard)

PerformanceGuard performanceGuardFromMap(const QVariantMap& map);
