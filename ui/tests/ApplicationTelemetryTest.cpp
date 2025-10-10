#include <QtTest/QtTest>
#include <QQmlApplicationEngine>
#include <QCommandLineParser>
#include <QFile>
#include <QIODevice>
#include <QScopeGuard>
#include <QGuiApplication>
#include <QQuickWindow>
#include <QScreen>
#include <QTemporaryDir>

#include "app/Application.hpp"
#include "telemetry/TelemetryReporter.hpp"
#include "telemetry/TelemetryTlsConfig.hpp"

#include <memory>
#include <optional>
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

    struct JankEvent {
        double frameMs = 0.0;
        double thresholdMs = 0.0;
        bool reduceMotion = false;
        int overlayActive = 0;
        int overlayAllowed = 0;
    };

    void setEnabled(bool enabled) override { m_enabled = enabled; }
    void setEndpoint(const QString& endpoint) override { m_endpoint = endpoint; }
    void setNotesTag(const QString& tag) override { m_tag = tag; }
    void setWindowCount(int count) override { m_windowCount = count; }
    void setTlsConfig(const TelemetryTlsConfig& config) override {
        m_tlsConfig = config;
        tlsConfigured = true;
    }
    void setAuthToken(const QString& token) override { authToken = token; }
    void setScreenInfo(const ScreenInfo& info) override { screenInfo = info; }
    void clearScreenInfo() override {
        screenInfo.reset();
        clearedScreen = true;
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

    void reportJankEvent(const PerformanceGuard& guard,
                         double frameTimeMs,
                         double thresholdMs,
                         bool reduceMotionActive,
                         int overlayActive,
                         int overlayAllowed) override {
        Q_UNUSED(guard);
        if (!m_enabled)
            return;
        JankEvent event;
        event.frameMs = frameTimeMs;
        event.thresholdMs = thresholdMs;
        event.reduceMotion = reduceMotionActive;
        event.overlayActive = overlayActive;
        event.overlayAllowed = overlayAllowed;
        jankEvents.push_back(event);
    }

    bool m_enabled = false;
    QString m_endpoint;
    QString m_tag;
    int m_windowCount = 0;
    TelemetryTlsConfig m_tlsConfig;
    bool tlsConfigured = false;
    QString authToken;
    std::optional<ScreenInfo> screenInfo;
    bool clearedScreen = false;
    std::vector<ReduceMotionEvent> reduceMotionEvents;
    std::vector<OverlayEvent> overlayEvents;
    std::vector<JankEvent> jankEvents;
};

class ApplicationTelemetryTest : public QObject {
    Q_OBJECT

private slots:
    void testReduceMotionEventDispatch();
    void testOverlayBudgetDispatch();
    void testTlsConfigurationForwarding();
    void testEnvironmentOverrides();
    void testJankTelemetryDispatch();
    void testPreferredScreenSelectionCli();
    void testPreferredScreenSelectionEnvironment();
    void testScreenMetadataForwarding();
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
        QStringLiteral("--reduce-motion-after"),
        QStringLiteral("0.15"),
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    app.notifyWindowCount(3);
    QCOMPARE(reporterPtr->m_windowCount, 3);

    app.notifyOverlayUsage(2, 4, false);

    for (int i = 0; i < 3; ++i) {
        app.simulateFrameIntervalForTesting(0.016); // ~62 FPS
    }
    for (int i = 0; i < 4; ++i) {
        app.simulateFrameIntervalForTesting(0.12); // ~8 FPS, wymusi reduce motion
    }

    QCOMPARE(reporterPtr->reduceMotionEvents.size(), std::size_t{1});
    const auto& event = reporterPtr->reduceMotionEvents.front();
    QCOMPARE(event.active, true);
    QCOMPARE(event.overlayActive, 2);
    QCOMPARE(event.overlayAllowed, 4);
    QVERIFY(event.fps > 0.0);
    QVERIFY(event.fps < 20.0);

    // Ponowne zgłoszenie tej samej wartości nie powinno dublować wpisu.
    app.simulateFrameIntervalForTesting(0.11);
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

void ApplicationTelemetryTest::testJankTelemetryDispatch() {
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
        QStringLiteral("--jank-threshold-ms"),
        QStringLiteral("12.5"),
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    app.notifyOverlayUsage(1, 3, false);

    app.simulateFrameIntervalForTesting(0.05); // 50 ms klatka
    QCOMPARE(reporterPtr->jankEvents.size(), std::size_t{1});
    const auto& event = reporterPtr->jankEvents.front();
    QVERIFY(event.frameMs > 40.0);
    QCOMPARE(event.thresholdMs, 12.5);
    QCOMPARE(event.overlayActive, 1);
    QCOMPARE(event.overlayAllowed, 3);
    QCOMPARE(event.reduceMotion, false);

    // Kolejna próbka przed upływem cooldownu jest ignorowana
    app.simulateFrameIntervalForTesting(0.05);
    QCOMPARE(reporterPtr->jankEvents.size(), std::size_t{1});

    QTest::qWait(450);
    app.simulateFrameIntervalForTesting(0.05);
    QCOMPARE(reporterPtr->jankEvents.size(), std::size_t{2});
}

void ApplicationTelemetryTest::testScreenMetadataForwarding() {
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

    QQuickWindow window;
    window.resize(640, 480);
    if (QScreen* primary = QGuiApplication::primaryScreen()) {
        window.setScreen(primary);
    }
    app.applyPreferredScreenForTesting(&window);

    QVERIFY(reporterPtr->screenInfo.has_value());
    const auto& info = reporterPtr->screenInfo.value();
    QVERIFY(!info.name.isEmpty());
    QVERIFY(info.index >= -1);

    app.applyPreferredScreenForTesting(nullptr);
    QVERIFY(!reporterPtr->screenInfo.has_value());
    QVERIFY(reporterPtr->clearedScreen);
}

void ApplicationTelemetryTest::testPreferredScreenSelectionCli()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);

    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--metrics-endpoint"),
        QStringLiteral("dummy:5000"),
        QStringLiteral("--screen-index"),
        QStringLiteral("0"),
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    QQuickWindow window;
    window.resize(320, 200);
    app.applyPreferredScreenForTesting(&window);

    auto* expected = QGuiApplication::primaryScreen();
    QVERIFY(expected != nullptr);
    QCOMPARE(window.screen(), expected);
}

void ApplicationTelemetryTest::testPreferredScreenSelectionEnvironment()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    QCommandLineParser parser;
    app.configureParser(parser);

    auto restorePrimary = qScopeGuard([]() {
        qunsetenv("BOT_CORE_UI_SCREEN_PRIMARY");
        qunsetenv("BOT_CORE_UI_SCREEN_INDEX");
        qunsetenv("BOT_CORE_UI_SCREEN_NAME");
    });

    qputenv("BOT_CORE_UI_SCREEN_PRIMARY", QByteArrayLiteral("true"));
    qputenv("BOT_CORE_UI_SCREEN_NAME", QByteArrayLiteral("ignored"));
    qputenv("BOT_CORE_UI_SCREEN_INDEX", QByteArrayLiteral("5"));

    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--metrics-endpoint"),
        QStringLiteral("dummy:5000"),
    };
    parser.process(args);
    QVERIFY(app.applyParser(parser));

    QQuickWindow window;
    window.resize(400, 240);
    app.applyPreferredScreenForTesting(&window);

    auto* expected = QGuiApplication::primaryScreen();
    QVERIFY(expected != nullptr);
    QCOMPARE(window.screen(), expected);

    auto* resolved = app.pickPreferredScreenForTesting();
    QCOMPARE(resolved, expected);
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

void ApplicationTelemetryTest::testEnvironmentOverrides() {
    QQmlApplicationEngine engine;
    Application app(engine);

    auto reporter = std::make_unique<FakeTelemetryReporter>();
    FakeTelemetryReporter* reporterPtr = reporter.get();
    app.setTelemetryReporter(std::move(reporter));

    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());
    const QString tokenPath = tempDir.filePath(QStringLiteral("token.txt"));
    QFile tokenFile(tokenPath);
    QVERIFY(tokenFile.open(QIODevice::WriteOnly | QIODevice::Text));
    tokenFile.write("env-token\n");
    tokenFile.close();

    qputenv("BOT_CORE_UI_METRICS_ENDPOINT", QByteArray("env-host:6001"));
    qputenv("BOT_CORE_UI_METRICS_TAG", QByteArray("env-tag"));
    qputenv("BOT_CORE_UI_METRICS_ENABLED", QByteArray("false"));
    qputenv("BOT_CORE_UI_METRICS_ROOT_CERT", tokenPath.toUtf8());
    qputenv("BOT_CORE_UI_METRICS_AUTH_TOKEN_FILE", tokenPath.toUtf8());
    const auto envGuard = qScopeGuard([] {
        qunsetenv("BOT_CORE_UI_METRICS_ENDPOINT");
        qunsetenv("BOT_CORE_UI_METRICS_TAG");
        qunsetenv("BOT_CORE_UI_METRICS_ENABLED");
        qunsetenv("BOT_CORE_UI_METRICS_ROOT_CERT");
        qunsetenv("BOT_CORE_UI_METRICS_AUTH_TOKEN_FILE");
    });

    QCommandLineParser parser;
    app.configureParser(parser);
    parser.process(QStringList{QStringLiteral("test")});
    QVERIFY(app.applyParser(parser));

    QCOMPARE(reporterPtr->m_endpoint, QStringLiteral("env-host:6001"));
    QCOMPARE(reporterPtr->m_tag, QStringLiteral("env-tag"));
    QCOMPARE(reporterPtr->m_enabled, false);
    QVERIFY(reporterPtr->tlsConfigured);
    QCOMPARE(reporterPtr->m_tlsConfig.enabled, true);
    QCOMPARE(reporterPtr->m_tlsConfig.rootCertificatePath, tokenPath);
    QCOMPARE(reporterPtr->authToken, QStringLiteral("env-token"));

}

QTEST_MAIN(ApplicationTelemetryTest)
#include "ApplicationTelemetryTest.moc"

