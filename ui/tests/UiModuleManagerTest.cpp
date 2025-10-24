#include <QtTest/QtTest>
#include <QObject>
#include <QVariantList>
#include <QVariantMap>
#include <QUrl>

#include "app/UiModuleManager.hpp"

class UiModuleManagerTest : public QObject {
    Q_OBJECT

private slots:
    void registersViews();
    void registersServices();
    void handlesMissingPluginDirectories();
    void registersModules();
};

void UiModuleManagerTest::registersViews()
{
    UiModuleManager manager;
    UiModuleManager::ViewDescriptor dashboard{
        QStringLiteral("dashboard"),
        QStringLiteral("Dashboard"),
        QUrl(QStringLiteral("qrc:/qml/components/Dashboard.qml")),
        QStringLiteral("core"),
        { { QStringLiteral("category"), QStringLiteral("main") } },
    };

    QVERIFY(manager.registerView(QStringLiteral("core"), dashboard));
    QCOMPARE(manager.availableViews().size(), 1);

    // Duplikaty są blokowane
    QVERIFY(!manager.registerView(QStringLiteral("core"), dashboard));

    const QVariantList views = manager.availableViews();
    QCOMPARE(views.size(), 1);
    const QVariantMap viewMap = views.first().toMap();
    QCOMPARE(viewMap.value(QStringLiteral("id")).toString(), QStringLiteral("dashboard"));
    QCOMPARE(viewMap.value(QStringLiteral("moduleId")).toString(), QStringLiteral("core"));
    QCOMPARE(viewMap.value(QStringLiteral("name")).toString(), QStringLiteral("Dashboard"));
}

void UiModuleManagerTest::registersServices()
{
    UiModuleManager manager;

    bool singletonCreated = false;
    UiModuleManager::ServiceDescriptor singletonService;
    singletonService.id = QStringLiteral("telemetry");
    singletonService.name = QStringLiteral("TelemetryService");
    singletonService.factory = [&](QObject* parent) -> QObject* {
        singletonCreated = true;
        auto* object = new QObject(parent);
        object->setObjectName(QStringLiteral("telemetry-service"));
        return object;
    };

    QVERIFY(manager.registerService(QStringLiteral("core"), singletonService));
    QObject* firstInstance = manager.resolveService(QStringLiteral("telemetry"));
    QVERIFY(firstInstance);
    QVERIFY(singletonCreated);
    QCOMPARE(firstInstance->objectName(), QStringLiteral("telemetry-service"));
    QObject* secondInstance = manager.resolveService(QStringLiteral("telemetry"));
    QCOMPARE(firstInstance, secondInstance);

    UiModuleManager::ServiceDescriptor transientService;
    transientService.id = QStringLiteral("transient");
    transientService.name = QStringLiteral("TransientService");
    transientService.singleton = false;
    transientService.factory = [](QObject* parent) -> QObject* {
        auto* object = new QObject(parent);
        object->setObjectName(QStringLiteral("transient"));
        return object;
    };

    QVERIFY(manager.registerService(QStringLiteral("core"), transientService));
    QObject* transientA = manager.resolveService(QStringLiteral("transient"));
    QObject* transientB = manager.resolveService(QStringLiteral("transient"));
    QVERIFY(transientA);
    QVERIFY(transientB);
    QVERIFY(transientA != transientB);
}

void UiModuleManagerTest::handlesMissingPluginDirectories()
{
    UiModuleManager manager;
    manager.setPluginPaths({QStringLiteral("/definitely/missing/path")});
    QVERIFY(!manager.loadPlugins());
    QCOMPARE(manager.availableViews().size(), 0);
}

namespace {

class StubModule : public QObject, public UiModuleInterface {
public:
    QString moduleId() const override { return QStringLiteral("stub"); }

    void registerComponents(UiModuleManager& manager) override
    {
        UiModuleManager::ViewDescriptor view;
        view.id = QStringLiteral("stub.view");
        view.name = QStringLiteral("Stub View");
        view.source = QUrl(QStringLiteral("qrc:/stub.qml"));
        manager.registerView(moduleId(), view);

        UiModuleManager::ServiceDescriptor service;
        service.id = QStringLiteral("stub.service");
        service.name = QStringLiteral("Stub Service");
        service.factory = [](QObject* parent) -> QObject* {
            auto* object = new QObject(parent);
            object->setObjectName(QStringLiteral("stub-service"));
            return object;
        };
        manager.registerService(moduleId(), service);
    }
};

} // namespace

void UiModuleManagerTest::registersModules()
{
    UiModuleManager manager;
    StubModule module;
    manager.registerModule(&module);

    const QVariantList views = manager.availableViews();
    QCOMPARE(views.size(), 1);
    const QVariantMap descriptor = views.first().toMap();
    QCOMPARE(descriptor.value(QStringLiteral("id")).toString(), QStringLiteral("stub.view"));
    QCOMPARE(descriptor.value(QStringLiteral("moduleId")).toString(), QStringLiteral("stub"));

    QObject* service = manager.resolveService(QStringLiteral("stub.service"));
    QVERIFY(service);
    QCOMPARE(service->objectName(), QStringLiteral("stub-service"));
}

QTEST_MAIN(UiModuleManagerTest)
#include "UiModuleManagerTest.moc"
