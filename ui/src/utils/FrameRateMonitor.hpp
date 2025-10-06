#pragma once

#include <QObject>
#include <QElapsedTimer>

#include "PerformanceGuard.hpp"

class QQuickWindow;

class FrameRateMonitor : public QObject {
    Q_OBJECT
public:
    explicit FrameRateMonitor(QObject* parent = nullptr);

    void setWindow(QQuickWindow* window);
    void setPerformanceGuard(const PerformanceGuard& guard);
    void reset();

signals:
    void reduceMotionSuggested(bool enabled);

private slots:
    void handleFrameSwapped();

private:
    void applyReduceMotion(bool enabled);
    double lowFpsThreshold() const;

    QQuickWindow* m_window = nullptr;
    PerformanceGuard m_guard;
    QMetaObject::Connection m_connection;
    QElapsedTimer m_timer;
    bool m_timerValid = false;
    double m_lowFpsDuration = 0.0;
    double m_highFpsDuration = 0.0;
    bool m_reduceMotionActive = false;
};
