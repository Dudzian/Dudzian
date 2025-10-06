#include "PerformanceGuard.hpp"

PerformanceGuard performanceGuardFromMap(const QVariantMap& map) {
    PerformanceGuard guard;
    if (map.contains("fps_target")) {
        guard.fpsTarget = map.value("fps_target").toInt();
    }
    if (map.contains("reduce_motion_after_seconds")) {
        guard.reduceMotionAfterSeconds = map.value("reduce_motion_after_seconds").toDouble();
    }
    if (map.contains("jank_threshold_ms")) {
        guard.jankThresholdMs = map.value("jank_threshold_ms").toDouble();
    }
    if (map.contains("max_overlay_count")) {
        guard.maxOverlayCount = map.value("max_overlay_count").toInt();
    }
    return guard;
}
