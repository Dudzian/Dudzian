#pragma once

#include <QObject>
#include <QPointer>
#include <QTimer>

#include "telemetry/TelemetryReporter.hpp"
#include "utils/PerformanceGuard.hpp"

class FrameRateMonitor;
class UiTelemetryReporter;

class PerformanceTelemetryController : public QObject {
    Q_OBJECT
public:
    explicit PerformanceTelemetryController(QObject* parent = nullptr);

    void setTelemetryReporter(TelemetryReporter* reporter);
    void setFrameRateMonitor(FrameRateMonitor* monitor);
    void setPerformanceGuard(const PerformanceGuard& guard);

    void recordSystemMetrics(double cpuUtil, double gpuUtil, double ramMb, quint64 droppedFrames, quint64 processedPerSecond);

public slots:
    void handleFrameSample(double fps);

private:
    void publishSnapshot(double fps);

    void ensurePublishTimer();

    QPointer<FrameRateMonitor>  m_monitor;
    QPointer<UiTelemetryReporter> m_uiReporter;
    TelemetryReporter*          m_reporter = nullptr;
    PerformanceGuard            m_guard{};
    double                      m_lastCpu = 0.0;
    double                      m_lastGpu = 0.0;
    double                      m_lastRam = 0.0;
    quint64                     m_lastDropped = 0;
    quint64                     m_lastProcessed = 0;
    QTimer                      m_publishTimer;
    double                      m_pendingFps = 0.0;
    bool                        m_hasPendingFps = false;
};

