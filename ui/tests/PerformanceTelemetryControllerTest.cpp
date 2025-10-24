#include <QtTest>
#include <QJsonDocument>
#include <QJsonObject>

#include "telemetry/PerformanceTelemetryController.hpp"
#include "telemetry/UiTelemetryReporter.hpp"
#include "grpc/MetricsClient.hpp"

class RecordingMetricsClient final : public MetricsClientInterface {
public:
    void setEndpoint(const QString&) override {}
    void setTlsConfig(const TelemetryTlsConfig&) override {}
    void setAuthToken(const QString&) override {}
    void setRbacRole(const QString&) override {}

    bool pushSnapshot(const botcore::trading::v1::MetricsSnapshot& snapshot, QString* errorMessage = nullptr) override
    {
        Q_UNUSED(errorMessage);
        lastSnapshot = snapshot;
        calls += 1;
        return true;
    }

    int calls = 0;
    botcore::trading::v1::MetricsSnapshot lastSnapshot;
};

class PerformanceTelemetryControllerTest : public QObject {
    Q_OBJECT
private slots:
    void publishesSnapshots();
};

void PerformanceTelemetryControllerTest::publishesSnapshots()
{
    UiTelemetryReporter reporter;
    reporter.setEnabled(true);
    reporter.setEndpoint(QStringLiteral("in-memory"));

    auto metricsClient = std::make_shared<RecordingMetricsClient>();
    reporter.setMetricsClientForTesting(metricsClient);

    PerformanceTelemetryController controller;
    controller.setTelemetryReporter(&reporter);

    PerformanceGuard guard;
    guard.fpsTarget = 120;
    guard.maxOverlayCount = 3;
    controller.setPerformanceGuard(guard);
    controller.recordSystemMetrics(42.0, 18.0, 512.0, 2, 1200);

    controller.handleFrameSample(118.5);

    QCOMPARE(metricsClient->calls, 1);
    const auto& snapshot = metricsClient->lastSnapshot;
    QVERIFY(snapshot.has_fps());
    QCOMPARE(snapshot.fps(), 118.5);
    const QJsonObject json = QJsonDocument::fromJson(QString::fromStdString(snapshot.notes()).toUtf8()).object();
    QCOMPARE(json.value(QStringLiteral("cpu_util")).toDouble(), 42.0);
    QCOMPARE(json.value(QStringLiteral("gpu_util")).toDouble(), 18.0);
    QCOMPARE(json.value(QStringLiteral("dropped_frames")).toDouble(), 2.0);
    QCOMPARE(json.value(QStringLiteral("processed_per_second")).toDouble(), 1200.0);
    QCOMPARE(json.value(QStringLiteral("overlay_limit")).toInt(), 3);
}

QTEST_MAIN(PerformanceTelemetryControllerTest)
#include "PerformanceTelemetryControllerTest.moc"
