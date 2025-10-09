#include <QtTest>

#include "utils/FrameRateMonitor.hpp"
#include "utils/PerformanceGuard.hpp"

class FrameRateMonitorTest : public QObject {
    Q_OBJECT
private slots:
    void reducesMotionAfterLowFps();
    void recoversAfterHighFps();
};

void FrameRateMonitorTest::reducesMotionAfterLowFps() {
    FrameRateMonitor monitor;
    PerformanceGuard guard;
    guard.fpsTarget = 60;
    guard.reduceMotionAfterSeconds = 0.2; // przyspieszony próg do testów
    monitor.setPerformanceGuard(guard);

    QList<bool> events;
    QObject::connect(&monitor, &FrameRateMonitor::reduceMotionSuggested, [&events](bool enabled) {
        events.append(enabled);
    });

    // trzy klatki po 0.15 s -> 0.45 s poniżej progu, powinno włączyć reduce motion
    monitor.simulateFrameIntervalForTest(0.15);
    monitor.simulateFrameIntervalForTest(0.15);
    monitor.simulateFrameIntervalForTest(0.15);

    QVERIFY(!events.isEmpty());
    QCOMPARE(events.last(), true);
}

void FrameRateMonitorTest::recoversAfterHighFps() {
    FrameRateMonitor monitor;
    PerformanceGuard guard;
    guard.fpsTarget = 60;
    guard.reduceMotionAfterSeconds = 0.2;
    monitor.setPerformanceGuard(guard);

    QList<bool> events;
    QObject::connect(&monitor, &FrameRateMonitor::reduceMotionSuggested, [&events](bool enabled) {
        events.append(enabled);
    });

    // aktywuj reduce motion
    for (int i = 0; i < 4; ++i) {
        monitor.simulateFrameIntervalForTest(0.15);
    }
    QVERIFY(!events.isEmpty());
    QCOMPARE(events.last(), true);

    // kilkanaście próbek szybkich klatek -> powrót
    for (int i = 0; i < 16; ++i) {
        monitor.simulateFrameIntervalForTest(0.01); // ~100 FPS
    }

    QVERIFY(events.last() == false);
    QVERIFY(events.contains(false));
}

QTEST_MAIN(FrameRateMonitorTest)
#include "FrameRateMonitorTest.moc"
