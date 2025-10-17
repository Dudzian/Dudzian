#include <QtTest/QtTest>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickItem>
#include <QQuickWindow>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QIODevice>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "security/SecurityAdminController.hpp"

class AdminDialogE2ETest final : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void testAssignAndRemoveProfileFlow();

private:
    QString locateRepoRoot() const;
    QString m_repoRoot;
};

void AdminDialogE2ETest::initTestCase()
{
    qputenv("QT_QUICK_CONTROLS_STYLE", QByteArrayLiteral("Basic"));
    qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
    m_repoRoot = locateRepoRoot();
    QVERIFY2(!m_repoRoot.isEmpty(), "Nie znaleziono katalogu repozytorium dla PYTHONPATH");

    const QByteArray existing = qgetenv("PYTHONPATH");
    QByteArray updated = m_repoRoot.toUtf8();
    if (!existing.isEmpty()) {
        updated = existing + ':' + updated;
    }
    qputenv("PYTHONPATH", updated);
}

QString AdminDialogE2ETest::locateRepoRoot() const
{
    QDir dir(QCoreApplication::applicationDirPath());
    for (int depth = 0; depth < 12; ++depth) {
        if (dir.exists("bot_core") && dir.exists("ui")) {
            return dir.absolutePath();
        }
        if (!dir.cdUp()) {
            break;
        }
    }
    return QString();
}

void AdminDialogE2ETest::testAssignAndRemoveProfileFlow()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString licensePath = tempDir.filePath(QStringLiteral("license.json"));
    const QString profilesPath = tempDir.filePath(QStringLiteral("profiles.json"));
    const QString logPath = tempDir.filePath(QStringLiteral("security_admin.log"));

    QJsonObject licenseObject{{QStringLiteral("fingerprint"), QStringLiteral("OEM-XYZ-123")}};
    licenseObject.insert(QStringLiteral("valid"), QJsonObject{{QStringLiteral("from"), QStringLiteral("2024-01-01T00:00:00Z")},
                                                          {QStringLiteral("to"), QStringLiteral("2025-01-01T00:00:00Z")}});
    QFile licenseFile(licensePath);
    QVERIFY(licenseFile.open(QIODevice::WriteOnly | QIODevice::Text));
    licenseFile.write(QJsonDocument(licenseObject).toJson(QJsonDocument::Compact));
    licenseFile.close();

    QJsonArray profilesArray;
    profilesArray.append(QJsonObject{{QStringLiteral("user_id"), QStringLiteral("ops")},
                                     {QStringLiteral("display_name"), QStringLiteral("Operations")},
                                     {QStringLiteral("roles"), QJsonArray{QStringLiteral("metrics.read")}}});
    QFile profilesFile(profilesPath);
    QVERIFY(profilesFile.open(QIODevice::WriteOnly | QIODevice::Text));
    profilesFile.write(QJsonDocument(profilesArray).toJson(QJsonDocument::Compact));
    profilesFile.close();

    SecurityAdminController controller;
    controller.setLicensePath(licensePath);
    controller.setProfilesPath(profilesPath);
    controller.setLogPath(logPath);

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty(QStringLiteral("securityController"), &controller);
    const QUrl dialogUrl(QStringLiteral("qrc:/qml/components/AdminDialog.qml"));
    engine.load(dialogUrl);
    QVERIFY2(!engine.rootObjects().isEmpty(), "Nie udało się załadować AdminDialog.qml");

    QObject* dialog = engine.rootObjects().first();
    QVERIFY(dialog);

    QSignalSpy profilesSpy(&controller, &SecurityAdminController::userProfilesChanged);
    QSignalSpy busySpy(&controller, &SecurityAdminController::busyChanged);

    QMetaObject::invokeMethod(dialog, "open");
    QTRY_VERIFY_WITH_TIMEOUT(!controller.isBusy(), 5000);
    const int initialSpyCount = profilesSpy.count();

    QObject* statusLabel = dialog->findChild<QObject*>(QStringLiteral("licenseStatusValue"));
    QVERIFY(statusLabel);
    QTRY_COMPARE(statusLabel->property("text").toString(), QStringLiteral("active"));

    QObject* fingerprintLabel = dialog->findChild<QObject*>(QStringLiteral("licenseFingerprintValue"));
    QVERIFY(fingerprintLabel);
    QTRY_COMPARE(fingerprintLabel->property("text").toString(), QStringLiteral("OEM-XYZ-123"));

    QObject* listView = dialog->findChild<QObject*>(QStringLiteral("profilesView"));
    QVERIFY(listView);
    QTRY_COMPARE(listView->property("count").toInt(), 1);

    QObject* userIdField = dialog->findChild<QObject*>(QStringLiteral("userIdField"));
    QObject* displayNameField = dialog->findChild<QObject*>(QStringLiteral("displayNameField"));
    QObject* rolesField = dialog->findChild<QObject*>(QStringLiteral("rolesField"));
    QObject* saveButton = dialog->findChild<QObject*>(QStringLiteral("saveProfileButton"));
    QObject* removeButton = dialog->findChild<QObject*>(QStringLiteral("removeProfileButton"));

    QVERIFY(userIdField);
    QVERIFY(displayNameField);
    QVERIFY(rolesField);
    QVERIFY(saveButton);
    QVERIFY(removeButton);

    userIdField->setProperty("text", QStringLiteral("carol"));
    displayNameField->setProperty("text", QStringLiteral("Carol"));
    rolesField->setProperty("text", QStringLiteral("metrics.write, metrics.read"));

    QMetaObject::invokeMethod(saveButton, "click");

    QTRY_VERIFY_WITH_TIMEOUT(profilesSpy.count() > initialSpyCount, 5000);
    QTRY_VERIFY_WITH_TIMEOUT(!controller.isBusy(), 5000);
    QTRY_COMPARE(listView->property("count").toInt(), 2);

    const QVariantList profilesModel = controller.userProfiles();
    QCOMPARE(profilesModel.size(), 2);
    const QVariantMap newProfile = profilesModel.last().toMap();
    QCOMPARE(newProfile.value(QStringLiteral("user_id")).toString(), QStringLiteral("carol"));
    QCOMPARE(newProfile.value(QStringLiteral("display_name")).toString(), QStringLiteral("Carol"));
    const QVariantList rolesList = newProfile.value(QStringLiteral("roles")).toList();
    QVERIFY(rolesList.contains(QStringLiteral("metrics.read")));
    QVERIFY(rolesList.contains(QStringLiteral("metrics.write")));

    QFile verifyProfiles(profilesPath);
    QVERIFY(verifyProfiles.open(QIODevice::ReadOnly | QIODevice::Text));
    const QJsonDocument storedDoc = QJsonDocument::fromJson(verifyProfiles.readAll());
    verifyProfiles.close();
    QVERIFY(storedDoc.isArray());
    const QJsonArray storedArray = storedDoc.array();
    QCOMPARE(storedArray.size(), 2);

    QFile logFile(logPath);
    QVERIFY(logFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QByteArray logContent = logFile.readAll();
    QVERIFY2(logContent.contains("carol"), "Log administratora nie zawiera wpisu dla nowego profilu");
    logFile.close();

    // Kontroler powinien wyczyścić pola formularza.
    QCOMPARE(userIdField->property("text").toString(), QString());
    QCOMPARE(displayNameField->property("text").toString(), QString());
    QCOMPARE(rolesField->property("text").toString(), QString());

    const int afterAssignCount = profilesSpy.count();

    dialog->setProperty("selectedProfileId", QStringLiteral("ops"));
    QVERIFY(removeButton->property("enabled").toBool());
    QMetaObject::invokeMethod(removeButton, "click");

    QTRY_VERIFY_WITH_TIMEOUT(profilesSpy.count() > afterAssignCount, 5000);
    QTRY_VERIFY_WITH_TIMEOUT(!controller.isBusy(), 5000);
    QTRY_COMPARE(listView->property("count").toInt(), 1);

    const QVariantList updatedProfiles = controller.userProfiles();
    QCOMPARE(updatedProfiles.size(), 1);
    const QVariantMap remainingProfile = updatedProfiles.first().toMap();
    QCOMPARE(remainingProfile.value(QStringLiteral("user_id")).toString(), QStringLiteral("carol"));

    QFile updatedProfilesFile(profilesPath);
    QVERIFY(updatedProfilesFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QJsonDocument updatedDoc = QJsonDocument::fromJson(updatedProfilesFile.readAll());
    updatedProfilesFile.close();
    QVERIFY(updatedDoc.isArray());
    const QJsonArray updatedArray = updatedDoc.array();
    QCOMPARE(updatedArray.size(), 1);
    QCOMPARE(updatedArray.first().toObject().value(QStringLiteral("user_id")).toString(), QStringLiteral("carol"));

    Q_UNUSED(busySpy);
}

QTEST_MAIN(AdminDialogE2ETest)
#include "AdminDialogE2ETest.moc"

