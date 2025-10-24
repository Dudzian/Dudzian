#include <QtTest/QtTest>

#include <QVariantMap>
#include <QStringList>

#include "app/UiModuleManager.hpp"
#include "app/UiModuleServicesModel.hpp"

class UiModuleServicesModelTest : public QObject {
    Q_OBJECT

private slots:
    void reflectsRegisteredServices();
    void filtersBySearchQuery();
    void removesServicesOnUnregister();
};

void UiModuleServicesModelTest::reflectsRegisteredServices()
{
    UiModuleManager manager;
    UiModuleServicesModel model;
    model.setModuleManager(&manager);

    UiModuleManager::ServiceDescriptor telemetry;
    telemetry.id = QStringLiteral("telemetry");
    telemetry.name = QStringLiteral("TelemetryService");
    telemetry.metadata.insert(QStringLiteral("tags"), QStringList{QStringLiteral("metrics")});
    telemetry.factory = [](QObject* parent) -> QObject* {
        auto* object = new QObject(parent);
        object->setObjectName(QStringLiteral("telemetry"));
        return object;
    };

    UiModuleManager::ServiceDescriptor support;
    support.id = QStringLiteral("support");
    support.name = QStringLiteral("SupportToolsService");
    support.singleton = false;
    support.metadata.insert(QStringLiteral("owner"), QStringLiteral("Support"));
    support.factory = [](QObject* parent) -> QObject* {
        auto* object = new QObject(parent);
        object->setObjectName(QStringLiteral("support"));
        return object;
    };

    QVERIFY(manager.registerService(QStringLiteral("core.metrics"), telemetry));
    QVERIFY(manager.registerService(QStringLiteral("operations"), support));

    QCOMPARE(model.rowCount(), 2);

    const QVariantMap first = model.serviceAt(0);
    QCOMPARE(first.value(QStringLiteral("id")).toString(), QStringLiteral("support"));
    QCOMPARE(first.value(QStringLiteral("singleton")).toBool(), false);

    const QVariantMap lookup = model.findById(QStringLiteral("telemetry"));
    QCOMPARE(lookup.value(QStringLiteral("moduleId")).toString(), QStringLiteral("core.metrics"));
}

void UiModuleServicesModelTest::filtersBySearchQuery()
{
    UiModuleManager manager;
    UiModuleServicesModel model;
    model.setModuleManager(&manager);

    UiModuleManager::ServiceDescriptor diagnostics;
    diagnostics.id = QStringLiteral("diagnostics");
    diagnostics.name = QStringLiteral("DiagnosticsService");
    diagnostics.metadata.insert(QStringLiteral("category"), QStringLiteral("health"));
    diagnostics.factory = [](QObject* parent) -> QObject* {
        return new QObject(parent);
    };

    UiModuleManager::ServiceDescriptor streaming;
    streaming.id = QStringLiteral("streaming");
    streaming.name = QStringLiteral("StreamingBridge");
    streaming.metadata.insert(QStringLiteral("channel"), QStringLiteral("market"));
    streaming.factory = [](QObject* parent) -> QObject* {
        return new QObject(parent);
    };

    QVERIFY(manager.registerService(QStringLiteral("core.health"), diagnostics));
    QVERIFY(manager.registerService(QStringLiteral("market"), streaming));

    model.setSearchFilter(QStringLiteral("market"));
    QCOMPARE(model.rowCount(), 1);
    QCOMPARE(model.serviceAt(0).value(QStringLiteral("id")).toString(), QStringLiteral("streaming"));

    model.setSearchFilter(QStringLiteral("health"));
    QCOMPARE(model.rowCount(), 1);
    QCOMPARE(model.serviceAt(0).value(QStringLiteral("id")).toString(), QStringLiteral("diagnostics"));

    model.setSearchFilter(QStringLiteral("bridge"));
    QCOMPARE(model.rowCount(), 1);
    QCOMPARE(model.serviceAt(0).value(QStringLiteral("id")).toString(), QStringLiteral("streaming"));

    model.setSearchFilter(QStringLiteral("unknown"));
    QCOMPARE(model.rowCount(), 0);
}

void UiModuleServicesModelTest::removesServicesOnUnregister()
{
    UiModuleManager manager;
    UiModuleServicesModel model;
    model.setModuleManager(&manager);

    UiModuleManager::ServiceDescriptor transient;
    transient.id = QStringLiteral("transient");
    transient.name = QStringLiteral("TransientService");
    transient.singleton = false;
    transient.factory = [](QObject* parent) -> QObject* {
        return new QObject(parent);
    };

    QVERIFY(manager.registerService(QStringLiteral("core"), transient));
    QCOMPARE(model.rowCount(), 1);

    QVERIFY(manager.unregisterService(QStringLiteral("transient")));
    QCOMPARE(model.rowCount(), 0);
}

QTEST_MAIN(UiModuleServicesModelTest)
#include "UiModuleServicesModelTest.moc"

