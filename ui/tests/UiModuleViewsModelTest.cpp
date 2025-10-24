#include <QtTest/QtTest>

#include <QVariantMap>
#include <QUrl>
#include <QStringList>

#include "app/UiModuleManager.hpp"
#include "app/UiModuleViewsModel.hpp"

class UiModuleViewsModelTest : public QObject {
    Q_OBJECT

private slots:
    void reflectsRegisteredViews();
    void filtersByCategory();
    void removesViewsOnUnregister();
    void listsCategories();
};

void UiModuleViewsModelTest::reflectsRegisteredViews()
{
    UiModuleManager manager;
    UiModuleViewsModel model;
    model.setModuleManager(&manager);

    UiModuleManager::ViewDescriptor viewA;
    viewA.id = QStringLiteral("alpha");
    viewA.name = QStringLiteral("Alpha View");
    viewA.source = QUrl(QStringLiteral("qrc:/alpha.qml"));
    viewA.category = QStringLiteral("main");
    QVERIFY(manager.registerView(QStringLiteral("core"), viewA));

    QCOMPARE(model.rowCount(), 1);
    const QVariantMap row = model.viewAt(0);
    QCOMPARE(row.value(QStringLiteral("id")).toString(), QStringLiteral("alpha"));
    QCOMPARE(row.value(QStringLiteral("moduleId")).toString(), QStringLiteral("core"));
    QCOMPARE(row.value(QStringLiteral("category")).toString(), QStringLiteral("main"));

    const QVariantMap lookup = model.findById(QStringLiteral("alpha"));
    QCOMPARE(lookup.value(QStringLiteral("name")).toString(), QStringLiteral("Alpha View"));
}

void UiModuleViewsModelTest::filtersByCategory()
{
    UiModuleManager manager;
    UiModuleViewsModel model;
    model.setModuleManager(&manager);

    UiModuleManager::ViewDescriptor viewA;
    viewA.id = QStringLiteral("alpha");
    viewA.name = QStringLiteral("Alpha");
    viewA.source = QUrl(QStringLiteral("qrc:/alpha.qml"));
    viewA.category = QStringLiteral("main");
    QVERIFY(manager.registerView(QStringLiteral("core"), viewA));

    UiModuleManager::ViewDescriptor viewB;
    viewB.id = QStringLiteral("beta");
    viewB.name = QStringLiteral("Beta");
    viewB.source = QUrl(QStringLiteral("qrc:/beta.qml"));
    viewB.category = QStringLiteral("aux");
    QVERIFY(manager.registerView(QStringLiteral("core"), viewB));

    QCOMPARE(model.rowCount(), 2);

    model.setCategoryFilter(QStringLiteral("main"));
    QCOMPARE(model.rowCount(), 1);
    QCOMPARE(model.viewAt(0).value(QStringLiteral("id")).toString(), QStringLiteral("alpha"));

    model.setCategoryFilter(QString());
    QCOMPARE(model.rowCount(), 2);
    QCOMPARE(model.viewAt(0).value(QStringLiteral("name")).toString(), QStringLiteral("Alpha"));
    QCOMPARE(model.viewAt(1).value(QStringLiteral("name")).toString(), QStringLiteral("Beta"));
}

void UiModuleViewsModelTest::removesViewsOnUnregister()
{
    UiModuleManager manager;
    UiModuleViewsModel model;
    model.setModuleManager(&manager);

    UiModuleManager::ViewDescriptor view;
    view.id = QStringLiteral("alpha");
    view.name = QStringLiteral("Alpha");
    view.source = QUrl(QStringLiteral("qrc:/alpha.qml"));
    view.category = QStringLiteral("main");
    QVERIFY(manager.registerView(QStringLiteral("core"), view));
    QCOMPARE(model.rowCount(), 1);

    QVERIFY(manager.unregisterView(QStringLiteral("alpha")));
    QCOMPARE(model.rowCount(), 0);
}

void UiModuleViewsModelTest::listsCategories()
{
    UiModuleManager manager;
    UiModuleViewsModel model;
    model.setModuleManager(&manager);

    UiModuleManager::ViewDescriptor analytics;
    analytics.id = QStringLiteral("analytics");
    analytics.name = QStringLiteral("Analytics");
    analytics.source = QUrl(QStringLiteral("qrc:/analytics.qml"));
    analytics.category = QStringLiteral("analytics");
    QVERIFY(manager.registerView(QStringLiteral("core"), analytics));

    UiModuleManager::ViewDescriptor diagnostics;
    diagnostics.id = QStringLiteral("diagnostics");
    diagnostics.name = QStringLiteral("Diagnostics");
    diagnostics.source = QUrl(QStringLiteral("qrc:/diagnostics.qml"));
    diagnostics.category = QStringLiteral("diagnostics");
    QVERIFY(manager.registerView(QStringLiteral("core"), diagnostics));

    UiModuleManager::ViewDescriptor duplicateCategory;
    duplicateCategory.id = QStringLiteral("duplicate");
    duplicateCategory.name = QStringLiteral("Duplicate");
    duplicateCategory.source = QUrl(QStringLiteral("qrc:/duplicate.qml"));
    duplicateCategory.category = QStringLiteral("analytics");
    QVERIFY(manager.registerView(QStringLiteral("core"), duplicateCategory));

    UiModuleManager::ViewDescriptor withoutCategory;
    withoutCategory.id = QStringLiteral("generic");
    withoutCategory.name = QStringLiteral("Generic");
    withoutCategory.source = QUrl(QStringLiteral("qrc:/generic.qml"));
    QVERIFY(manager.registerView(QStringLiteral("core"), withoutCategory));

    const QStringList categories = model.categories();
    QCOMPARE(categories.size(), 2);
    QCOMPARE(categories.at(0), QStringLiteral("analytics"));
    QCOMPARE(categories.at(1), QStringLiteral("diagnostics"));

    QVERIFY(manager.unregisterView(QStringLiteral("diagnostics")));
    const QStringList reduced = model.categories();
    QCOMPARE(reduced.size(), 1);
    QCOMPARE(reduced.first(), QStringLiteral("analytics"));
}

QTEST_MAIN(UiModuleViewsModelTest)
#include "UiModuleViewsModelTest.moc"

