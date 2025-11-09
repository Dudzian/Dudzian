#include "PerformanceGuard.hpp"

#include <QVariant>
#include <QVariantMap>

/**
 * @brief Buduje strukturę PerformanceGuard z mapy (np. z QML / YAML).
 *
 * Obsługiwane klucze (bieżący format):
 *  - "fps_target" (int)
 *  - "reduce_motion_after_seconds" (double)
 *  - "jank_threshold_ms" (double)
 *  - "max_overlay_count" (int)
 *  - "disable_secondary_when_fps_below" (int)
 *
 * Wsteczna kompatybilność (alternatywny format):
 *  performance_guard:
 *    overlays:
 *      max_overlays: <int>
 *      disable_secondary_when_fps_below: <int>
 */
PerformanceGuard performanceGuardFromMap(const QVariantMap& map) {
    PerformanceGuard guard;

    // Aktualne klucze
    if (map.contains(QStringLiteral("fps_target"))) {
        guard.fpsTarget = map.value(QStringLiteral("fps_target")).toInt();
    }
    if (map.contains(QStringLiteral("reduce_motion_after_seconds"))) {
        guard.reduceMotionAfterSeconds = map.value(QStringLiteral("reduce_motion_after_seconds")).toDouble();
    }
    if (map.contains(QStringLiteral("jank_threshold_ms"))) {
        guard.jankThresholdMs = map.value(QStringLiteral("jank_threshold_ms")).toDouble();
    }
    if (map.contains(QStringLiteral("max_overlay_count"))) {
        guard.maxOverlayCount = map.value(QStringLiteral("max_overlay_count")).toInt();
    }
    if (map.contains(QStringLiteral("disable_secondary_when_fps_below"))) {
        guard.disableSecondaryWhenFpsBelow = map.value(QStringLiteral("disable_secondary_when_fps_below")).toInt();
    }

    // Wsteczna kompatybilność: performance_guard.overlays.{max_overlays, disable_secondary_when_fps_below}
    if (map.contains(QStringLiteral("overlays"))) {
        const auto overlays = map.value(QStringLiteral("overlays")).toMap();
        if (overlays.contains(QStringLiteral("max_overlays"))) {
            guard.maxOverlayCount = overlays.value(QStringLiteral("max_overlays")).toInt();
        }
        if (overlays.contains(QStringLiteral("disable_secondary_when_fps_below"))) {
            guard.disableSecondaryWhenFpsBelow = overlays.value(QStringLiteral("disable_secondary_when_fps_below")).toInt();
        }
    }

    return guard;
}

QVariantMap performanceGuardToMap(const PerformanceGuard& guard)
{
    QVariantMap map;
    map.insert(QStringLiteral("fps_target"), guard.fpsTarget);
    map.insert(QStringLiteral("reduce_motion_after_seconds"), guard.reduceMotionAfterSeconds);
    map.insert(QStringLiteral("jank_threshold_ms"), guard.jankThresholdMs);
    map.insert(QStringLiteral("max_overlay_count"), guard.maxOverlayCount);
    map.insert(QStringLiteral("disable_secondary_when_fps_below"), guard.disableSecondaryWhenFpsBelow);

    return map;
}
