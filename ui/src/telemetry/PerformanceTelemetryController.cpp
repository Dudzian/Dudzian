#include "PerformanceTelemetryController.hpp"

#include <QJsonObject>
#include <QMetaObject>

#include "telemetry/UiTelemetryReporter.hpp"
#include "utils/FrameRateMonitor.hpp"

PerformanceTelemetryController::PerformanceTelemetryController(QObject* parent)
    : QObject(parent)
{
}

void PerformanceTelemetryController::setTelemetryReporter(TelemetryReporter* reporter)
{
    m_reporter = reporter;
    m_uiReporter = qobject_cast<UiTelemetryReporter*>(reporter);
}

void PerformanceTelemetryController::setFrameRateMonitor(FrameRateMonitor* monitor)
{
    if (m_monitor == monitor) {
        return;
    }
    if (m_monitor) {
        disconnect(m_monitor, &FrameRateMonitor::frameSampled, this, &PerformanceTelemetryController::handleFrameSample);
    }
    m_monitor = monitor;
    if (m_monitor) {
        connect(m_monitor, &FrameRateMonitor::frameSampled, this, &PerformanceTelemetryController::handleFrameSample);
    }
}

void PerformanceTelemetryController::setPerformanceGuard(const PerformanceGuard& guard)
{
    m_guard = guard;
}

void PerformanceTelemetryController::recordSystemMetrics(double cpuUtil, double gpuUtil, double ramMb, quint64 droppedFrames, quint64 processedPerSecond)
{
    m_lastCpu = cpuUtil;
    m_lastGpu = gpuUtil;
    m_lastRam = ramMb;
    m_lastDropped = droppedFrames;
    m_lastProcessed = processedPerSecond;
}

void PerformanceTelemetryController::handleFrameSample(double fps)
{
    publishSnapshot(fps);
}

void PerformanceTelemetryController::publishSnapshot(double fps)
{
    if (!m_uiReporter) {
        return;
    }

    QJsonObject notes;
    notes.insert(QStringLiteral("fps_target"), m_guard.fpsTarget);
    notes.insert(QStringLiteral("reduce_motion_budget"), m_guard.reduceMotionSeconds);
    notes.insert(QStringLiteral("overlay_limit"), m_guard.maxOverlayCount);
    notes.insert(QStringLiteral("cpu_util"), m_lastCpu);
    notes.insert(QStringLiteral("gpu_util"), m_lastGpu);
    notes.insert(QStringLiteral("ram_mb"), m_lastRam);
    notes.insert(QStringLiteral("dropped_frames"), static_cast<double>(m_lastDropped));
    notes.insert(QStringLiteral("processed_per_second"), static_cast<double>(m_lastProcessed));

    m_uiReporter->pushSnapshot(notes, fps);
}

