#include "FrameRateMonitor.hpp"

#include <QQuickWindow>
#include <QtGlobal>

FrameRateMonitor::FrameRateMonitor(QObject* parent)
    : QObject(parent) {}

void FrameRateMonitor::setWindow(QQuickWindow* window) {
    if (window == m_window)
        return;

    if (m_window)
        disconnect(m_connection);

    m_window = window;
    m_timerValid = false;
    m_lowFpsDuration = 0.0;
    m_highFpsDuration = 0.0;

    if (!m_window)
        return;

    m_connection = connect(
        m_window, &QQuickWindow::frameSwapped,
        this, &FrameRateMonitor::handleFrameSwapped
    );
}

void FrameRateMonitor::setPerformanceGuard(const PerformanceGuard& guard) {
    m_guard = guard;
    reset();
}

void FrameRateMonitor::reset() {
    m_timerValid = false;
    m_lowFpsDuration = 0.0;
    m_highFpsDuration = 0.0;
    if (m_reduceMotionActive)
        applyReduceMotion(false);
}

void FrameRateMonitor::handleFrameSwapped() {
    if (!m_timerValid) {
        m_timer.start();
        m_timerValid = true;
        return;
    }

    const qint64 elapsedNs = m_timer.nsecsElapsed();
    m_timer.restart();
    if (elapsedNs <= 0)
        return;

    const double deltaSeconds = static_cast<double>(elapsedNs) / 1'000'000'000.0;
    if (deltaSeconds <= 0.0)
        return;

    const double fps = 1.0 / deltaSeconds;
    m_lastFps = fps;
    Q_EMIT frameSampled(fps);

    const double threshold = lowFpsThreshold();
    if (threshold <= 0.0)
        return;

    const double reduceSeconds = qMax(0.0, m_guard.reduceMotionAfterSeconds);

    if (fps < threshold) {
        m_lowFpsDuration += deltaSeconds;
        m_highFpsDuration = 0.0;

        if (!m_reduceMotionActive && (reduceSeconds == 0.0 || m_lowFpsDuration >= reduceSeconds))
            applyReduceMotion(true);
    } else {
        m_highFpsDuration += deltaSeconds;
        m_lowFpsDuration = 0.0;

        const double recoverySeconds =
            (reduceSeconds > 0.0) ? qMax(0.2, reduceSeconds * 0.5) : 0.3;

        if (m_reduceMotionActive && m_highFpsDuration >= recoverySeconds)
            applyReduceMotion(false);
    }
}

void FrameRateMonitor::applyReduceMotion(bool enabled) {
    if (m_reduceMotionActive == enabled)
        return;
    m_reduceMotionActive = enabled;
    Q_EMIT reduceMotionSuggested(m_reduceMotionActive);
}

double FrameRateMonitor::lowFpsThreshold() const {
    if (m_guard.disableSecondaryWhenFpsBelow > 0)
        return static_cast<double>(m_guard.disableSecondaryWhenFpsBelow);
    if (m_guard.fpsTarget >= 120)
        return 110.0;
    if (m_guard.fpsTarget >= 90)
        return static_cast<double>(m_guard.fpsTarget - 10);
    if (m_guard.fpsTarget >= 60)
        return 55.0;
    return qMax(30.0, static_cast<double>(m_guard.fpsTarget) * 0.9);
}
