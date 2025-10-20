#include <QtTest/QtTest>

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include "license/LicenseActivationController.hpp"

namespace {
QJsonDocument buildLicenseDocument(const QString& fingerprint)
{
    QJsonObject payload{
        {QStringLiteral("schema"), QStringLiteral("core.oem.license")},
        {QStringLiteral("schema_version"), QStringLiteral("1.0")},
        {QStringLiteral("fingerprint"), fingerprint},
        {QStringLiteral("issued_at"), QStringLiteral("2024-01-01T00:00:00Z")},
        {QStringLiteral("expires_at"), QStringLiteral("2099-12-31T00:00:00Z")},
        {QStringLiteral("profile"), QStringLiteral("paper")},
        {QStringLiteral("issuer"), QStringLiteral("OEM-Control")},
        {QStringLiteral("bundle_version"), QStringLiteral("2.3.1")},
        {QStringLiteral("features"), QJsonArray{QStringLiteral("daemon"), QStringLiteral("ui")}},
    };
    QJsonObject signature{
        {QStringLiteral("algorithm"), QStringLiteral("HMAC-SHA384")},
        {QStringLiteral("value"), QStringLiteral("dummy-signature")},
        {QStringLiteral("key_id"), QStringLiteral("signing-key-1")},
    };
    return QJsonDocument(QJsonObject{
        {QStringLiteral("payload"), payload},
        {QStringLiteral("signature"), signature},
    });
}

QString writeFingerprintDocument(const QString& dir, const QString& fingerprint)
{
    const QString path = QDir(dir).filePath(QStringLiteral("fingerprint.expected.json"));
    QJsonObject payload{{QStringLiteral("fingerprint"), fingerprint},
                        {QStringLiteral("generated_at"), QStringLiteral("2024-01-01T00:00:00Z")}};
    QJsonObject signature{{QStringLiteral("algorithm"), QStringLiteral("HMAC-SHA384")},
                           {QStringLiteral("value"), QStringLiteral("stub")}};
    QFile file(path);
    file.open(QIODevice::WriteOnly | QIODevice::Text);
    file.write(QJsonDocument(QJsonObject{{QStringLiteral("payload"), payload},
                                        {QStringLiteral("signature"), signature}})
                   .toJson(QJsonDocument::Indented));
    file.close();
    return path;
}

void writeLicenseFile(const QString& path, const QJsonDocument& doc)
{
    QFile file(path);
    QVERIFY2(file.open(QIODevice::WriteOnly | QIODevice::Text), "Unable to open license file for writing");
    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();
}

} // namespace

class LicenseActivationControllerTest : public QObject {
    Q_OBJECT

private slots:
    void activatesWithValidLicense();
    void rejectsMismatchedFingerprint();
    void acceptsBase64Payload();
    void autoProvisionImportsMatchingLicense();
    void autoProvisionRunsDuringInitialize();
};

void LicenseActivationControllerTest::activatesWithValidLicense()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString configDir = tempDir.filePath(QStringLiteral("config"));
    QDir().mkpath(configDir);
    const QString licenseDir = tempDir.filePath(QStringLiteral("var/licenses/active"));
    QDir().mkpath(licenseDir);

    const QString expectedFingerprint = QStringLiteral("DEVICE-12345");
    writeFingerprintDocument(configDir, expectedFingerprint);

    const QString licenseFile = tempDir.filePath(QStringLiteral("license.json"));
    writeLicenseFile(licenseFile, buildLicenseDocument(expectedFingerprint));

    LicenseActivationController controller;
    controller.setConfigDirectory(configDir);
    controller.setLicenseStoragePath(QDir(licenseDir).filePath(QStringLiteral("license.json")));
    controller.initialize();

    QVERIFY(controller.loadLicenseFile(licenseFile));
    QVERIFY(controller.licenseActive());
    QCOMPARE(controller.licenseFingerprint(), expectedFingerprint);
    QCOMPARE(controller.licenseProfile(), QStringLiteral("paper"));
    QCOMPARE(controller.licenseFeatures(), QStringList({QStringLiteral("daemon"), QStringLiteral("ui")}));

    QFile persisted(controller.licenseStoragePath());
    QVERIFY(persisted.exists());
}

void LicenseActivationControllerTest::rejectsMismatchedFingerprint()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString configDir = tempDir.filePath(QStringLiteral("config"));
    QDir().mkpath(configDir);
    writeFingerprintDocument(configDir, QStringLiteral("DEVICE-AAA"));

    LicenseActivationController controller;
    controller.setConfigDirectory(configDir);
    controller.setLicenseStoragePath(tempDir.filePath(QStringLiteral("var/licenses/active/license.json")));
    controller.initialize();

    const QJsonDocument doc = buildLicenseDocument(QStringLiteral("DEVICE-BBB"));
    const QString payload = QString::fromUtf8(doc.toJson(QJsonDocument::Compact));

    QVERIFY(!controller.applyLicenseText(payload));
    QVERIFY(!controller.licenseActive());
    QVERIFY(controller.statusIsError());
    QVERIFY(controller.statusMessage().contains(QStringLiteral("Fingerprint")));

    QFile persisted(controller.licenseStoragePath());
    QVERIFY(!persisted.exists());
}

void LicenseActivationControllerTest::acceptsBase64Payload()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString configDir = tempDir.filePath(QStringLiteral("config"));
    QDir().mkpath(configDir);
    const QString fingerprint = QStringLiteral("DEVICE-999");
    writeFingerprintDocument(configDir, fingerprint);

    LicenseActivationController controller;
    controller.setConfigDirectory(configDir);
    controller.setLicenseStoragePath(tempDir.filePath(QStringLiteral("var/licenses/active/license.json")));
    controller.initialize();

    const QByteArray encoded = buildLicenseDocument(fingerprint)
                                   .toJson(QJsonDocument::Compact)
                                   .toBase64();

    QVERIFY(controller.applyLicenseText(QString::fromUtf8(encoded)));
    QVERIFY(controller.licenseActive());
    QCOMPARE(controller.licenseFingerprint(), fingerprint);
    QVERIFY(controller.statusMessage().contains(QStringLiteral("Licencja aktywna")));
}

void LicenseActivationControllerTest::autoProvisionImportsMatchingLicense()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString configDir = tempDir.filePath(QStringLiteral("config"));
    QDir().mkpath(configDir);
    const QString inboxDir = tempDir.filePath(QStringLiteral("inbox"));
    QDir().mkpath(inboxDir);

    const QString fingerprint = QStringLiteral("AUTO-12345");
    const QString licenseFile = QDir(inboxDir).filePath(QStringLiteral("license.json"));
    writeLicenseFile(licenseFile, buildLicenseDocument(fingerprint));

    LicenseActivationController controller;
    controller.setConfigDirectory(configDir);
    controller.setProvisioningDirectory(inboxDir);
    controller.setLicenseStoragePath(tempDir.filePath(QStringLiteral("var/licenses/active/license.json")));
    controller.initialize();

    QVariantMap fingerprintDoc{{QStringLiteral("fingerprint"), fingerprint}};
    QVERIFY(controller.autoProvision(fingerprintDoc));
    QVERIFY(controller.licenseActive());
    QCOMPARE(controller.licenseFingerprint(), fingerprint);

    QFile archivedLicense(licenseFile + QStringLiteral(".applied"));
    QVERIFY(archivedLicense.exists());
}

void LicenseActivationControllerTest::autoProvisionRunsDuringInitialize()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString configDir = tempDir.filePath(QStringLiteral("config"));
    QDir().mkpath(configDir);
    const QString inboxDir = tempDir.filePath(QStringLiteral("inbox"));
    QDir().mkpath(inboxDir);

    const QString fingerprint = QStringLiteral("BOOT-INIT-001");
    writeFingerprintDocument(configDir, fingerprint);
    const QString licenseFile = QDir(inboxDir).filePath(QStringLiteral("license.json"));
    writeLicenseFile(licenseFile, buildLicenseDocument(fingerprint));

    LicenseActivationController controller;
    controller.setConfigDirectory(configDir);
    controller.setProvisioningDirectory(inboxDir);
    controller.setLicenseStoragePath(tempDir.filePath(QStringLiteral("var/licenses/active/license.json")));
    controller.initialize();

    QVERIFY(controller.licenseActive());
    QCOMPARE(controller.licenseFingerprint(), fingerprint);
    QFile archivedLicense(licenseFile + QStringLiteral(".applied"));
    QVERIFY(archivedLicense.exists());
}

QTEST_MAIN(LicenseActivationControllerTest)
#include "LicenseActivationControllerTest.moc"
