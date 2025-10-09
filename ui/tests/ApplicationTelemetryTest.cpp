#include <QtTest/QtTest>
#include <QQmlApplicationEngine>
#include <QCommandLineParser>

#include "app/Application.hpp"
#include "telemetry/TelemetryReporter.hpp"
#include "telemetry/TelemetryTlsConfig.hpp"

#include <memory>
#include <vector>

class FakeTelemetryReporter final : public TelemetryReporter {
public:
    struct ReduceMotionEvent {
        bool active = false;
        double fps = 0.0;
        int overlayActive = 0;
        int overlayAllowed = 0;
    };

    struct OverlayEvent {
        int active = 0;
        int allowed = 0;
        bool reduceMotion = false;
    };

    void setEnabled(bool enabled) override { m_enabled = enabled; }
    void setEndpoint(const QString& endpoint) override { m_endpoint = endpoint; }
    void setNotesTag(const QString& tag) override { m_tag = tag; }
    void setWindowCount(int count) override { m_windowCount = count; }
    void setTlsConfig(const TelemetryTlsConfig& config) override {
        m_tlsConfig = config;
        tlsConfigured = true;
    }
    bool isEnabled() const override { return m_enabled; }

    void reportReduceMotion(const PerformanceGuard& guard,
                            bool active,
                            double fps,
                            int overlayActive,
                            int overlayAllowed) override {
        Q_UNUSED(guard);
        if (!m_enabled)
            return;
        ReduceMotionEvent event;
        event.active = active;
        event.fps = fps;
        event.overlayActive = overlayActive;
        event.overlayAllowed = overlayAllowed;
        reduceMotionEvents.push_back(event);
    }

    void reportOverlayBudget(const PerformanceGuard& guard,
                             int overlayActive,
                             int overlayAllowed,
                             bool reduceMotionActive) override {
        Q_UNUSED(guard);
        if (!m_enabled)
            return;
        OverlayEvent event;
        event.active = overlayActive;
        event.allowed = overlayAllowed;
        event.reduceMotion = reduceMotionActive;
        overlayEvents.push_back(event);
    }

    bool m_enabled = false;
    QString m_endpoint;
    QString m_tag;
    int m_windowCount = 0;
    TelemetryTlsConfig m_tlsConfig;
    bool tlsConfigured = false;
    std::vector<ReduceMotionEvent> reduceMotionEvents;
    std::vector<OverlayEvent> overlayEvents;
};

class ApplicationTelemetryTest : public QObject {
    Q_OBJECT

private slots:
    void testReduceMotionEventDispatch();
    void testOverlayBudgetDispatch();
    void testTlsConfigurationForwarding();
};

void ApplicationTelemetryTest::testReduceMotionEventDispatch() {
    QQmlApplicationEngine engine;
    Application app(engine);

    auto reporter = std::make_unique<FakeTelemetryReporter>();
    FakeTelemetryReporter* reporterPtr = reporter.get();
    app.setTelemetryReporter(std::move(reporter));

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--metrics-endpoint"),
        QStringLiteral("dummy:5000"),
        QStringLiteral("--metrics-tag"),
        QStringLiteral("qt-test"),
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    app.notifyWindowCount(3);
    QCOMPARE(reporterPtr->m_windowCount, 3);

    app.ingestFpsSampleForTesting(48.5);
    app.notifyOverlayUsage(2, 4, false);
    app.setReduceMotionStateForTesting(true);

    QCOMPARE(reporterPtr->reduceMotionEvents.size(), std::size_t{1});
    const auto& event = reporterPtr->reduceMotionEvents.front();
    QCOMPARE(event.active, true);
    QCOMPARE(event.overlayActive, 2);
    QCOMPARE(event.overlayAllowed, 4);
    QCOMPARE(event.fps, 48.5);

    // Ponowne zgłoszenie tej samej wartości nie powinno dublować wpisu.
    app.setReduceMotionStateForTesting(true);
    QCOMPARE(reporterPtr->reduceMotionEvents.size(), std::size_t{1});
}

void ApplicationTelemetryTest::testOverlayBudgetDispatch() {
    QQmlApplicationEngine engine;
    Application app(engine);

    auto reporter = std::make_unique<FakeTelemetryReporter>();
    FakeTelemetryReporter* reporterPtr = reporter.get();
    app.setTelemetryReporter(std::move(reporter));

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--metrics-endpoint"),
        QStringLiteral("dummy:5000"),
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    app.ingestFpsSampleForTesting(59.0);
    app.notifyOverlayUsage(5, 3, true);

    QCOMPARE(reporterPtr->overlayEvents.size(), std::size_t{1});
    const auto& event = reporterPtr->overlayEvents.front();
    QCOMPARE(event.active, 5);
    QCOMPARE(event.allowed, 3);
    QCOMPARE(event.reduceMotion, true);

    // Ponowny stan bez zmian nie generuje kolejnego wpisu.
    app.notifyOverlayUsage(5, 3, true);
    QCOMPARE(reporterPtr->overlayEvents.size(), std::size_t{1});
}

void ApplicationTelemetryTest::testTlsConfigurationForwarding() {
    QQmlApplicationEngine engine;
    Application app(engine);

    auto reporter = std::make_unique<FakeTelemetryReporter>();
    FakeTelemetryReporter* reporterPtr = reporter.get();
    app.setTelemetryReporter(std::move(reporter));

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--metrics-endpoint"),
        QStringLiteral("secure:5000"),
        QStringLiteral("--metrics-use-tls"),
        QStringLiteral("--metrics-root-cert"),
        QStringLiteral("/tmp/root.pem"),
        QStringLiteral("--metrics-client-cert"),
        QStringLiteral("/tmp/client.pem"),
        QStringLiteral("--metrics-client-key"),
        QStringLiteral("/tmp/client.key"),
        QStringLiteral("--metrics-server-name"),
        QStringLiteral("metrics.internal"),
        QStringLiteral("--metrics-server-sha256"),
        QStringLiteral("deadbeef"),
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    QVERIFY(reporterPtr->tlsConfigured);
    QCOMPARE(reporterPtr->m_tlsConfig.enabled, true);
    QCOMPARE(reporterPtr->m_tlsConfig.rootCertificatePath, QStringLiteral("/tmp/root.pem"));
    QCOMPARE(reporterPtr->m_tlsConfig.clientCertificatePath, QStringLiteral("/tmp/client.pem"));
    QCOMPARE(reporterPtr->m_tlsConfig.clientKeyPath, QStringLiteral("/tmp/client.key"));
    QCOMPARE(reporterPtr->m_tlsConfig.serverNameOverride, QStringLiteral("metrics.internal"));
    QCOMPARE(reporterPtr->m_tlsConfig.pinnedServerSha256, QStringLiteral("deadbeef"));
}

QTEST_MAIN(ApplicationTelemetryTest)
#include "ApplicationTelemetryTest.moc"

