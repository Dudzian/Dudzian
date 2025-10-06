#include "PerformanceGuard.hpp"

#include <QVariant>
#include <QVariantMap>

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
    if (map.contains("disable_secondary_when_fps_below")) {
        guard.disableSecondaryWhenFpsBelow = map.value("disable_secondary_when_fps_below").toInt();
    }

    // Backwards/alt config format:
    // performance_guard:
    //   overlays:
    //     max_overlays: 4
    //     disable_secondary_when_fps_below: 55
    if (map.contains("overlays")) {
        const auto overlays = map.value("overlays").toMap();
        if (overlays.contains("max_overlays")) {
            guard.maxOverlayCount = overlays.value("max_overlays").toInt();
        }
        if (overlays.contains("disable_secondary_when_fps_below")) {
            guard.disableSecondaryWhenFpsBelow = overlays.value("disable_secondary_when_fps_below").toInt();
        }
    }

    return guard;
}
