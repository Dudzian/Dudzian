#include <QtTest/QtTest>

#include <QQmlApplicationEngine>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QScopeGuard>
#include <QTemporaryDir>
#include <QUrl>

#include "app/Application.hpp"
#include "app/UiModuleManager.hpp"
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

    UiModuleManager::ViewDescriptor view;
    view.id = QStringLiteral("stub.view");
    view.name = QStringLiteral("Stub View");
    view.source = QUrl(QStringLiteral("qrc:/stub.qml"));
    view.category = QStringLiteral("main");
    QVERIFY(recording->registerView(QStringLiteral("stub"), view));
    QCOMPARE(viewsModel->rowCount(), 1);
    QCOMPARE(viewsModel->viewAt(0).value(QStringLiteral("id")).toString(), QStringLiteral("stub.view"));
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

QTEST_MAIN(ApplicationUiModulesTest)
#include "ApplicationUiModulesTest.moc"

