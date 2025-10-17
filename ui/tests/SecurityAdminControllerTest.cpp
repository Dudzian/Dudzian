#include <QtTest/QtTest>
#include <QTemporaryDir>
#include <QFile>
#include <QIODevice>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QVariantList>

#include "security/SecurityAdminController.hpp"

class SecurityAdminControllerTest : public QObject {
    Q_OBJECT

private slots:
    void refreshLoadsState();
    void assignProfileUpdatesJson();
    void removeProfileDeletesJsonEntry();
};

namespace {

QString writeJsonFile(const QString& path, const QJsonObject& object)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QFAIL(qPrintable(QStringLiteral("Nie można zapisać pliku testowego %1: %2").arg(path, file.errorString())));
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Compact));
    file.close();
    return path;
}

} // namespace

void SecurityAdminControllerTest::refreshLoadsState()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString licensePath = tempDir.filePath(QStringLiteral("license.json"));
    const QString profilesPath = tempDir.filePath(QStringLiteral("profiles.json"));

    QJsonObject license{
        {QStringLiteral("fingerprint"), QStringLiteral("ABC-123")},
        {QStringLiteral("valid"), QJsonObject{{QStringLiteral("from"), QStringLiteral("2024-01-01T00:00:00Z")},
                                               {QStringLiteral("to"), QStringLiteral("2025-01-01T00:00:00Z")}}}
    };
    writeJsonFile(licensePath, license);

    QJsonObject profilesRoot;
    QJsonArray profilesArray;
    profilesArray.append(QJsonObject{{QStringLiteral("user_id"), QStringLiteral("ops")},
                                     {QStringLiteral("display_name"), QStringLiteral("Operations")},
                                     {QStringLiteral("roles"), QJsonArray{QStringLiteral("metrics.read")}}});
    profilesArray.append(QJsonObject{{QStringLiteral("user_id"), QStringLiteral("qa")},
                                     {QStringLiteral("display_name"), QStringLiteral("QA")},
                                     {QStringLiteral("roles"), QJsonArray{QStringLiteral("metrics.write")}}});
    QFile profilesFile(profilesPath);
    QVERIFY(profilesFile.open(QIODevice::WriteOnly | QIODevice::Text));
    profilesFile.write(QJsonDocument(profilesArray).toJson(QJsonDocument::Compact));
    profilesFile.close();

    SecurityAdminController controller;
    controller.setLicensePath(licensePath);
    controller.setProfilesPath(profilesPath);
    controller.setLogPath(tempDir.filePath(QStringLiteral("admin.log")));

    QVERIFY(controller.refresh());

    const QVariantMap licenseInfo = controller.licenseInfo();
    QCOMPARE(licenseInfo.value(QStringLiteral("fingerprint")).toString(), QStringLiteral("ABC-123"));
    QCOMPARE(licenseInfo.value(QStringLiteral("status")).toString(), QStringLiteral("active"));

    const QVariantList userProfiles = controller.userProfiles();
    QCOMPARE(userProfiles.size(), 2);
    const QVariantMap firstProfile = userProfiles.first().toMap();
    QVERIFY(firstProfile.value(QStringLiteral("user_id")).toString().contains(QStringLiteral("ops")));
}

void SecurityAdminControllerTest::assignProfileUpdatesJson()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString profilesPath = tempDir.filePath(QStringLiteral("profiles.json"));
    QFile profilesFile(profilesPath);
    QVERIFY(profilesFile.open(QIODevice::WriteOnly | QIODevice::Text));
    profilesFile.write("[]");
    profilesFile.close();

    const QString licensePath = tempDir.filePath(QStringLiteral("license.json"));
    QFile licenseFile(licensePath);
    QVERIFY(licenseFile.open(QIODevice::WriteOnly | QIODevice::Text));
    licenseFile.write("{\"fingerprint\": \"XYZ\"}");
    licenseFile.close();

    SecurityAdminController controller;
    controller.setLicensePath(licensePath);
    controller.setProfilesPath(profilesPath);
    controller.setLogPath(tempDir.filePath(QStringLiteral("admin.log")));
    controller.refresh();

    QStringList roles;
    roles << QStringLiteral("metrics.read") << QStringLiteral("metrics.write");
    QVERIFY(controller.assignProfile(QStringLiteral("new-user"), roles, QStringLiteral("New User")));

    QFile verifyFile(profilesPath);
    QVERIFY(verifyFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QJsonDocument doc = QJsonDocument::fromJson(verifyFile.readAll());
    verifyFile.close();
    QVERIFY(doc.isArray());
    const QJsonArray array = doc.array();
    QCOMPARE(array.size(), 1);
    const QJsonObject obj = array.first().toObject();
    QCOMPARE(obj.value(QStringLiteral("user_id")).toString(), QStringLiteral("new-user"));
    QCOMPARE(obj.value(QStringLiteral("display_name")).toString(), QStringLiteral("New User"));

    QFile logFile(tempDir.filePath(QStringLiteral("admin.log")));
    QVERIFY(logFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QByteArray logData = logFile.readAll();
    QVERIFY(logData.contains("new-user"));

    const QVariantList profiles = controller.userProfiles();
    QCOMPARE(profiles.size(), 1);
    const QVariantMap profileMap = profiles.first().toMap();
    QCOMPARE(profileMap.value(QStringLiteral("user_id")).toString(), QStringLiteral("new-user"));
}

void SecurityAdminControllerTest::removeProfileDeletesJsonEntry()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString profilesPath = tempDir.filePath(QStringLiteral("profiles.json"));
    QJsonArray profilesArray;
    profilesArray.append(QJsonObject{{QStringLiteral("user_id"), QStringLiteral("alice")},
                                     {QStringLiteral("display_name"), QStringLiteral("Alice")},
                                     {QStringLiteral("roles"), QJsonArray{QStringLiteral("metrics.read")}}});
    profilesArray.append(QJsonObject{{QStringLiteral("user_id"), QStringLiteral("bob")},
                                     {QStringLiteral("display_name"), QStringLiteral("Bob")},
                                     {QStringLiteral("roles"), QJsonArray{QStringLiteral("metrics.write")}}});
    QFile profilesFile(profilesPath);
    QVERIFY(profilesFile.open(QIODevice::WriteOnly | QIODevice::Text));
    profilesFile.write(QJsonDocument(profilesArray).toJson(QJsonDocument::Compact));
    profilesFile.close();

    SecurityAdminController controller;
    controller.setProfilesPath(profilesPath);
    controller.setLogPath(tempDir.filePath(QStringLiteral("admin.log")));
    controller.refresh();

    QVERIFY(controller.removeProfile(QStringLiteral("bob")));

    QFile verifyFile(profilesPath);
    QVERIFY(verifyFile.open(QIODevice::ReadOnly | QIODevice::Text));
    const QJsonDocument doc = QJsonDocument::fromJson(verifyFile.readAll());
    verifyFile.close();
    QVERIFY(doc.isArray());
    const QJsonArray updatedArray = doc.array();
    QCOMPARE(updatedArray.size(), 1);
    QCOMPARE(updatedArray.first().toObject().value(QStringLiteral("user_id")).toString(), QStringLiteral("alice"));
}

QTEST_MAIN(SecurityAdminControllerTest)
#include "SecurityAdminControllerTest.moc"

