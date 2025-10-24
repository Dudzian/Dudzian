#include <QtTest/QtTest>

#include <QQmlApplicationEngine>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QScopeGuard>
#include <QTemporaryDir>
#include <QUrl>
#include <QSignalSpy>

#include "app/Application.hpp"
#include "app/UiModuleManager.hpp"
#include "app/UiModuleServicesModel.hpp"
#include "app/UiModuleViewsModel.hpp"

namespace {

class RecordingModuleManager : public UiModuleManager {
public:
    using UiModuleManager::UiModuleManager;

    bool loadPlugins(const QStringList& candidates = {}) override
    {
        ++loadCallCount;
        lastCandidates = candidates;
        recordedPluginPaths = UiModuleManager::pluginPaths();
        LoadReport report;
        report.requestedPaths = candidates.isEmpty() ? recordedPluginPaths : candidates;
        report.loadedPlugins = recordedPluginPaths;
        report.pluginsLoaded = 0;
        report.viewsRegistered = 0;
        report.servicesRegistered = 0;
        setLastLoadReportForTesting(report);
        return loadResult;
    }

    int loadCallCount = 0;
    QStringList lastCandidates;
    QStringList recordedPluginPaths;
    bool loadResult = true;
};

QString absolutePath(const QString& path)
{
    return QFileInfo(path).absoluteFilePath();
}

void restoreEnv(const QByteArray& key, const QByteArray& original)
{
    if (original.isNull())
        qunsetenv(key.constData());
    else
        qputenv(key.constData(), original);
}

} // namespace

class ApplicationUiModulesTest : public QObject {
    Q_OBJECT

private slots:
    void cliOverridesModuleDirectories();
    void environmentProvidesDirectories();
    void fallbackUsesDefaultDirectories();
    void reloadsModulesOnDemand();
    void managesModuleDirectoriesAtRuntime();
    void autoReloadTriggeredByWatcher();
    void autoReloadWatchesMissingDirectories();
};

void ApplicationUiModulesTest::cliOverridesModuleDirectories()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto moduleManager = std::make_unique<RecordingModuleManager>();
    RecordingModuleManager* recording = moduleManager.get();
    app.setModuleManagerForTesting(std::move(moduleManager));

    QTemporaryDir dirA;
    QTemporaryDir dirB;
    QVERIFY(dirA.isValid());
    QVERIFY(dirB.isValid());

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--metrics-endpoint"), QStringLiteral("localhost:9000"),
        QStringLiteral("--ui-module-dir"), dirA.path(),
        QStringLiteral("--ui-module-dir"), dirA.path() + QLatin1Char('/'),
        QStringLiteral("--ui-module-dir"), dirB.path(),
    };
    parser.process(args);

    QVERIFY(app.applyParser(parser));

    const QStringList expected{
        absolutePath(dirA.path()),
        absolutePath(dirB.path()),
    };

    QCOMPARE(recording->pluginPaths(), expected);
    QCOMPARE(app.uiModuleDirectoriesForTesting(), expected);
    QCOMPARE(recording->loadCallCount, 1);
    QCOMPARE(recording->lastCandidates, QStringList());

    auto* viewsModel = app.moduleViewsModelForTesting();
    QVERIFY(viewsModel);
    QCOMPARE(viewsModel->rowCount(), 0);
    auto* servicesModel = app.moduleServicesModelForTesting();
    QVERIFY(servicesModel);
    QCOMPARE(servicesModel->rowCount(), 0);

    UiModuleManager::ViewDescriptor view;
    view.id = QStringLiteral("stub.view");
    view.name = QStringLiteral("Stub View");
    view.source = QUrl(QStringLiteral("qrc:/stub.qml"));
    view.category = QStringLiteral("main");
    QVERIFY(recording->registerView(QStringLiteral("stub"), view));
    QCOMPARE(viewsModel->rowCount(), 1);
    QCOMPARE(viewsModel->viewAt(0).value(QStringLiteral("id")).toString(), QStringLiteral("stub.view"));

    UiModuleManager::ServiceDescriptor service;
    service.id = QStringLiteral("stub.service");
    service.name = QStringLiteral("StubService");
    service.factory = [](QObject* parent) -> QObject* { return new QObject(parent); };
    QVERIFY(recording->registerService(QStringLiteral("stub"), service));
    QCOMPARE(servicesModel->rowCount(), 1);
    QCOMPARE(servicesModel->serviceAt(0).value(QStringLiteral("id")).toString(), QStringLiteral("stub.service"));
}

void ApplicationUiModulesTest::environmentProvidesDirectories()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto moduleManager = std::make_unique<RecordingModuleManager>();
    RecordingModuleManager* recording = moduleManager.get();
    app.setModuleManagerForTesting(std::move(moduleManager));

    QTemporaryDir dirA;
    QTemporaryDir dirB;
    QVERIFY(dirA.isValid());
    QVERIFY(dirB.isValid());

    const QByteArray original = qgetenv("BOT_CORE_UI_MODULE_DIRS");
    const QString envValue = dirA.path() + QDir::listSeparator() + dirB.path();
    qputenv("BOT_CORE_UI_MODULE_DIRS", envValue.toUtf8());
    const auto cleanup = qScopeGuard([&]() { restoreEnv(QByteArrayLiteral("BOT_CORE_UI_MODULE_DIRS"), original); });

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--metrics-endpoint"), QStringLiteral("localhost:9000"),
    };
    parser.process(args);

    QVERIFY(app.applyParser(parser));

    const QStringList expected{
        absolutePath(dirA.path()),
        absolutePath(dirB.path()),
    };

    QCOMPARE(recording->pluginPaths(), expected);
    QCOMPARE(app.uiModuleDirectoriesForTesting(), expected);
    QCOMPARE(recording->loadCallCount, 1);
    QCOMPARE(recording->lastCandidates, QStringList());
}

void ApplicationUiModulesTest::fallbackUsesDefaultDirectories()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto moduleManager = std::make_unique<RecordingModuleManager>();
    RecordingModuleManager* recording = moduleManager.get();
    app.setModuleManagerForTesting(std::move(moduleManager));

    const QByteArray original = qgetenv("BOT_CORE_UI_MODULE_DIRS");
    const auto cleanup = qScopeGuard([&]() { restoreEnv(QByteArrayLiteral("BOT_CORE_UI_MODULE_DIRS"), original); });
    qunsetenv("BOT_CORE_UI_MODULE_DIRS");

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--metrics-endpoint"), QStringLiteral("localhost:9000"),
    };
    parser.process(args);

    QVERIFY(app.applyParser(parser));

    const QString binaryModules = absolutePath(QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("modules")));
    const QString repoModules = absolutePath(QDir::current().absoluteFilePath(QStringLiteral("ui/modules")));

    QStringList expected;
    if (!binaryModules.isEmpty())
        expected.append(binaryModules);
    if (!repoModules.isEmpty() && !expected.contains(repoModules))
        expected.append(repoModules);

    QCOMPARE(recording->pluginPaths(), expected);
    QCOMPARE(app.uiModuleDirectoriesForTesting(), expected);
    QCOMPARE(recording->loadCallCount, expected.isEmpty() ? 0 : 1);
}

void ApplicationUiModulesTest::reloadsModulesOnDemand()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto moduleManager = std::make_unique<RecordingModuleManager>();
    RecordingModuleManager* recording = moduleManager.get();
    app.setModuleManagerForTesting(std::move(moduleManager));

    QTemporaryDir dirA;
    QVERIFY(dirA.isValid());

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--metrics-endpoint"), QStringLiteral("localhost:9000"),
        QStringLiteral("--ui-module-dir"), dirA.path(),
    };
    parser.process(args);

    QVERIFY(app.applyParser(parser));

    const QStringList expected{ absolutePath(dirA.path()) };
    QCOMPARE(recording->loadCallCount, 1);
    QCOMPARE(recording->pluginPaths(), expected);
    QCOMPARE(app.uiModuleDirectoriesForTesting(), expected);

    QSignalSpy reloadSpy(&app, &Application::uiModulesReloaded);
    QVERIFY(reloadSpy.isValid());

    recording->loadResult = true;
    QVERIFY(app.reloadUiModules());
    QCOMPARE(recording->loadCallCount, 2);
    QCOMPARE(recording->recordedPluginPaths, expected);
    QCOMPARE(recording->lastCandidates, QStringList());
    QCOMPARE(reloadSpy.count(), 1);
    {
        const QList<QVariant> arguments = reloadSpy.takeFirst();
        QCOMPARE(arguments.size(), 2);
        QCOMPARE(arguments.at(0).toBool(), true);
        const QVariantMap report = arguments.at(1).toMap();
        QCOMPARE(report.value(QStringLiteral("requestedPaths")).toStringList(), expected);
    }

    recording->loadResult = false;
    QVERIFY(!app.reloadUiModules());
    QCOMPARE(recording->loadCallCount, 3);
    QCOMPARE(recording->recordedPluginPaths, expected);
    QCOMPARE(recording->lastCandidates, QStringList());
    QCOMPARE(reloadSpy.count(), 1);
    {
        const QList<QVariant> arguments = reloadSpy.takeFirst();
        QCOMPARE(arguments.size(), 2);
        QCOMPARE(arguments.at(0).toBool(), false);
        const QVariantMap report = arguments.at(1).toMap();
        QCOMPARE(report.value(QStringLiteral("requestedPaths")).toStringList(), expected);
    }
}

void ApplicationUiModulesTest::managesModuleDirectoriesAtRuntime()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto moduleManager = std::make_unique<RecordingModuleManager>();
    RecordingModuleManager* recording = moduleManager.get();
    app.setModuleManagerForTesting(std::move(moduleManager));

    QTemporaryDir dirA;
    QTemporaryDir dirB;
    QVERIFY(dirA.isValid());
    QVERIFY(dirB.isValid());

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--metrics-endpoint"), QStringLiteral("localhost:9000"),
        QStringLiteral("--ui-module-dir"), dirA.path(),
    };
    parser.process(args);

    QVERIFY(app.applyParser(parser));

    const QString normA = absolutePath(dirA.path());
    const QString normB = absolutePath(dirB.path());

    QStringList expected{ normA };
    QCOMPARE(app.uiModuleDirectoriesForTesting(), expected);
    QCOMPARE(recording->loadCallCount, 1);
    QCOMPARE(recording->recordedPluginPaths, expected);

    QSignalSpy reloadSpy(&app, &Application::uiModulesReloaded);
    QVERIFY(reloadSpy.isValid());
    reloadSpy.clear();

    const QString added = app.addUiModuleDirectory(dirB.path());
    QCOMPARE(added, normB);
    expected.append(normB);
    QCOMPARE(app.uiModuleDirectoriesForTesting(), expected);
    QCOMPARE(recording->loadCallCount, 2);
    QCOMPARE(recording->recordedPluginPaths, expected);
    QCOMPARE(reloadSpy.count(), 1);
    {
        const QList<QVariant> arguments = reloadSpy.takeFirst();
        QCOMPARE(arguments.size(), 2);
        QCOMPARE(arguments.at(0).toBool(), true);
        const QVariantMap report = arguments.at(1).toMap();
        QCOMPARE(report.value(QStringLiteral("requestedPaths")).toStringList(), expected);
    }

    QStringList watched = app.watchedUiModulePathsForTesting();
    QVERIFY(watched.contains(normA));
    QVERIFY(watched.contains(normB));

    reloadSpy.clear();
    QCOMPARE(app.addUiModuleDirectory(dirB.path()), QString());
    QCOMPARE(recording->loadCallCount, 2);
    QCOMPARE(reloadSpy.count(), 0);

    QVERIFY(!app.removeUiModuleDirectory(QStringLiteral("/nonexistent/path")));
    QCOMPARE(recording->loadCallCount, 2);

    reloadSpy.clear();
    QVERIFY(app.removeUiModuleDirectory(dirA.path()));
    expected = QStringList{ normB };
    QCOMPARE(app.uiModuleDirectoriesForTesting(), expected);
    QCOMPARE(recording->loadCallCount, 3);
    QCOMPARE(recording->recordedPluginPaths, expected);
    QCOMPARE(reloadSpy.count(), 1);
    {
        const QList<QVariant> arguments = reloadSpy.takeFirst();
        QCOMPARE(arguments.size(), 2);
        QCOMPARE(arguments.at(0).toBool(), true);
        const QVariantMap report = arguments.at(1).toMap();
        QCOMPARE(report.value(QStringLiteral("requestedPaths")).toStringList(), expected);
    }

    watched = app.watchedUiModulePathsForTesting();
    QVERIFY(!watched.contains(normA));
    QVERIFY(watched.contains(normB));

    reloadSpy.clear();
    QVERIFY(app.removeUiModuleDirectory(dirB.path()));
    QStringList empty;
    QCOMPARE(app.uiModuleDirectoriesForTesting(), empty);
    QCOMPARE(recording->loadCallCount, 4);
    QCOMPARE(recording->recordedPluginPaths, empty);
    QCOMPARE(reloadSpy.count(), 1);
    {
        const QList<QVariant> arguments = reloadSpy.takeFirst();
        QCOMPARE(arguments.size(), 2);
        QCOMPARE(arguments.at(0).toBool(), true);
        const QVariantMap report = arguments.at(1).toMap();
        QCOMPARE(report.value(QStringLiteral("requestedPaths")).toStringList(), empty);
        QCOMPARE(report.value(QStringLiteral("loadedPlugins")).toStringList(), empty);
    }

    watched = app.watchedUiModulePathsForTesting();
    QVERIFY(!watched.contains(normB));
    QVERIFY(watched.isEmpty());
}

void ApplicationUiModulesTest::autoReloadTriggeredByWatcher()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto moduleManager = std::make_unique<RecordingModuleManager>();
    RecordingModuleManager* recording = moduleManager.get();
    app.setModuleManagerForTesting(std::move(moduleManager));

    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--metrics-endpoint"), QStringLiteral("localhost:9000"),
        QStringLiteral("--ui-module-dir"), dir.path(),
    };
    parser.process(args);

    QVERIFY(app.applyParser(parser));

    const QString expectedDir = absolutePath(dir.path());
    const QStringList watched = app.watchedUiModulePathsForTesting();
    QVERIFY(watched.contains(expectedDir));

    QVERIFY(!app.uiModuleAutoReloadEnabled());
    QSignalSpy autoReloadSpy(&app, &Application::uiModuleAutoReloadEnabledChanged);
    QVERIFY(autoReloadSpy.isValid());
    app.setUiModuleAutoReloadEnabled(true);
    QCOMPARE(app.uiModuleAutoReloadEnabled(), true);
    QCOMPARE(autoReloadSpy.count(), 1);

    app.setUiModuleAutoReloadDebounceForTesting(5);

    QSignalSpy reloadSpy(&app, &Application::uiModulesReloaded);
    QVERIFY(reloadSpy.isValid());
    reloadSpy.clear();
    recording->loadResult = true;

    app.triggerUiModuleWatcherForTesting(expectedDir);
    QTRY_VERIFY_WITH_TIMEOUT(reloadSpy.count() > 0, 200);

    {
        const QList<QVariant> arguments = reloadSpy.takeFirst();
        QCOMPARE(arguments.size(), 2);
        QCOMPARE(arguments.at(0).toBool(), true);
    }

    reloadSpy.clear();
    app.setUiModuleAutoReloadEnabled(false);
    QCOMPARE(app.uiModuleAutoReloadEnabled(), false);
    app.triggerUiModuleWatcherForTesting(expectedDir);
    QTest::qWait(50);
    QCOMPARE(reloadSpy.count(), 0);
}

void ApplicationUiModulesTest::autoReloadWatchesMissingDirectories()
{
    QQmlApplicationEngine engine;
    Application app(engine);

    auto moduleManager = std::make_unique<RecordingModuleManager>();
    RecordingModuleManager* recording = moduleManager.get();
    app.setModuleManagerForTesting(std::move(moduleManager));

    QTemporaryDir rootDir;
    QVERIFY(rootDir.isValid());

    const QString missingDir = QDir(rootDir.path()).absoluteFilePath(QStringLiteral("modules/ui"));
    QVERIFY(!QFileInfo::exists(missingDir));

    QCommandLineParser parser;
    app.configureParser(parser);
    const QStringList args{
        QStringLiteral("test"),
        QStringLiteral("--endpoint"), QStringLiteral("localhost:50051"),
        QStringLiteral("--metrics-endpoint"), QStringLiteral("localhost:9000"),
        QStringLiteral("--ui-module-dir"), missingDir,
    };
    parser.process(args);

    QVERIFY(app.applyParser(parser));

    const QString parentDir = QFileInfo(rootDir.path()).absoluteFilePath();
    const QStringList watched = app.watchedUiModulePathsForTesting();
    QVERIFY(watched.contains(parentDir));

    QVERIFY(!app.uiModuleAutoReloadEnabled());
    app.setUiModuleAutoReloadEnabled(true);
    QCOMPARE(app.uiModuleAutoReloadEnabled(), true);

    app.setUiModuleAutoReloadDebounceForTesting(5);

    QSignalSpy reloadSpy(&app, &Application::uiModulesReloaded);
    QVERIFY(reloadSpy.isValid());
    reloadSpy.clear();

    recording->loadResult = true;
    app.triggerUiModuleWatcherForTesting(parentDir);
    QTRY_VERIFY_WITH_TIMEOUT(reloadSpy.count() > 0, 200);

    const QList<QVariant> arguments = reloadSpy.takeFirst();
    QCOMPARE(arguments.size(), 2);
    QCOMPARE(arguments.at(0).toBool(), true);
}

QTEST_MAIN(ApplicationUiModulesTest)
#include "ApplicationUiModulesTest.moc"

