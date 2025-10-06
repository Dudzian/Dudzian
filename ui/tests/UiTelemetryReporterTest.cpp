#include <QtTest/QtTest>
#include <QJsonDocument>
#include <QJsonObject>

#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/security/server_credentials.h>

#include "telemetry/UiTelemetryReporter.hpp"
#include "utils/PerformanceGuard.hpp"
#include "trading.grpc.pb.h"

#include <mutex>
#include <vector>

class MetricsCaptureService final : public botcore::trading::v1::MetricsService::Service {
public:
    grpc::Status StreamMetrics(grpc::ServerContext*, const botcore::trading::v1::StreamMetricsRequest*,
                               grpc::ServerWriter<botcore::trading::v1::MetricsSnapshot>*) override {
        return grpc::Status::OK;
    }

    grpc::Status PushMetrics(grpc::ServerContext*, const botcore::trading::v1::MetricsSnapshot* request,
                             botcore::trading::v1::MetricsAck* response) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_snapshots.push_back(*request);
        response->set_accepted(true);
        return grpc::Status::OK;
    }

    std::vector<botcore::trading::v1::MetricsSnapshot> takeSnapshots() {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto copy = m_snapshots;
        m_snapshots.clear();
        return copy;
    }

private:
    std::mutex m_mutex;
    std::vector<botcore::trading::v1::MetricsSnapshot> m_snapshots;
};

class UiTelemetryReporterTest final : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testReduceMotionEvent();
    void testOverlayBudgetEvent();

private:
    std::unique_ptr<grpc::Server> m_server;
    std::unique_ptr<MetricsCaptureService> m_service;
    QString m_address;
};

void UiTelemetryReporterTest::initTestCase() {
    m_service = std::make_unique<MetricsCaptureService>();
    grpc::ServerBuilder builder;
    int selectedPort = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selectedPort);
    builder.RegisterService(m_service.get());
    m_server = builder.BuildAndStart();
    QVERIFY2(m_server != nullptr, "Nie udało się uruchomić serwera testowego gRPC");
    QVERIFY(selectedPort != 0);
    m_address = QStringLiteral("127.0.0.1:%1").arg(selectedPort);
}

void UiTelemetryReporterTest::cleanupTestCase() {
    if (m_server) {
        m_server->Shutdown();
        m_server->Wait();
    }
}

void UiTelemetryReporterTest::testReduceMotionEvent() {
    UiTelemetryReporter reporter;
    reporter.setEndpoint(m_address);
    reporter.setEnabled(true);
    reporter.setNotesTag(QStringLiteral("test-tag"));
    reporter.setWindowCount(3);

    PerformanceGuard guard;
    guard.fpsTarget = 60;
    guard.jankThresholdMs = 18.0;
    guard.maxOverlayCount = 4;
    guard.disableSecondaryWhenFpsBelow = 45;

    reporter.reportReduceMotion(guard, true, 48.5, 2, 4);

    const auto snapshots = m_service->takeSnapshots();
    QCOMPARE(snapshots.size(), std::size_t{1});
    const auto& snapshot = snapshots.front();
    QCOMPARE(snapshot.fps(), 48.5);

    const auto notes = QString::fromStdString(snapshot.notes());
    const auto json = QJsonDocument::fromJson(notes.toUtf8()).object();
    QCOMPARE(json.value(QStringLiteral("event")).toString(), QStringLiteral("reduce_motion"));
    QCOMPARE(json.value(QStringLiteral("active")).toBool(), true);
    QCOMPARE(json.value(QStringLiteral("overlay_active")).toInt(), 2);
    QCOMPARE(json.value(QStringLiteral("overlay_allowed")).toInt(), 4);
    QCOMPARE(json.value(QStringLiteral("tag")).toString(), QStringLiteral("test-tag"));
    QCOMPARE(json.value(QStringLiteral("window_count")).toInt(), 3);
    QCOMPARE(json.value(QStringLiteral("disable_secondary_fps")).toInt(), 45);
}

void UiTelemetryReporterTest::testOverlayBudgetEvent() {
    UiTelemetryReporter reporter;
    reporter.setEndpoint(m_address);
    reporter.setEnabled(true);
    reporter.setNotesTag(QStringLiteral("overlay-test"));
    reporter.setWindowCount(2);

    PerformanceGuard guard;
    guard.fpsTarget = 120;
    guard.maxOverlayCount = 5;

    reporter.reportOverlayBudget(guard, 3, 5, false);

    const auto snapshots = m_service->takeSnapshots();
    QCOMPARE(snapshots.size(), std::size_t{1});
    const auto& snapshot = snapshots.front();
    QCOMPARE(snapshot.fps(), 0.0);

    const auto notes = QString::fromStdString(snapshot.notes());
    const auto json = QJsonDocument::fromJson(notes.toUtf8()).object();
    QCOMPARE(json.value(QStringLiteral("event")).toString(), QStringLiteral("overlay_budget"));
    QCOMPARE(json.value(QStringLiteral("active_overlays")).toInt(), 3);
    QCOMPARE(json.value(QStringLiteral("allowed_overlays")).toInt(), 5);
    QCOMPARE(json.value(QStringLiteral("reduce_motion")).toBool(), false);
    QCOMPARE(json.value(QStringLiteral("window_count")).toInt(), 2);
}

QTEST_MAIN(UiTelemetryReporterTest)
#include "UiTelemetryReporterTest.moc"
