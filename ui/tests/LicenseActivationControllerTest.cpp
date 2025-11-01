#include <QtTest/QtTest>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QSignalSpy>
#include <QTimer>

#include "license/LicenseActivationController.hpp"

namespace {
QJsonDocument buildLicenseDocument(const QString& fingerprint)
{
    QJsonObject payload{
        {QStringLiteral("license_id"), QStringLiteral("demo-pro")},
        {QStringLiteral("edition"), QStringLiteral("pro")},
        {QStringLiteral("issued_at"), QStringLiteral("2024-01-01")},
        {QStringLiteral("maintenance_until"), QStringLiteral("2099-12-31")},
        {QStringLiteral("environments"), QJsonArray{QStringLiteral("demo"), QStringLiteral("paper")}},
        {QStringLiteral("modules"), QJsonObject{{QStringLiteral("futures"), true}, {QStringLiteral("walk_forward"), false}}},
        {QStringLiteral("runtime"), QJsonObject{{QStringLiteral("auto_trader"), true}}},
        {QStringLiteral("holder"), QJsonObject{{QStringLiteral("name"), QStringLiteral("OEM-Control")},
                                                {QStringLiteral("email"), QStringLiteral("ops@example.com")}}},
        {QStringLiteral("seats"), 2},
        {QStringLiteral("hwid"), fingerprint},
    };
    const QByteArray payloadBytes = QJsonDocument(payload).toJson(QJsonDocument::Compact);
    QJsonObject bundle{
        {QStringLiteral("payload_b64"), QString::fromUtf8(payloadBytes.toBase64())},
        {QStringLiteral("signature_b64"), QStringLiteral("ZHVtbXk=")},
    };
    return QJsonDocument(bundle);
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
    QDir().mkpath(QFileInfo(path).dir().path());
    QFile file(path);
    QVERIFY2(file.open(QIODevice::WriteOnly | QIODevice::Text), "Unable to open license file for writing");
    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();
}

class MockBindingSecretJob : public LicenseActivationController::BindingSecretJob {
public:
    enum Mode {
        AutoSuccess,
        AutoFailure,
        Manual,
    };

    explicit MockBindingSecretJob(Mode mode, QObject* parent = nullptr)
        : LicenseActivationController::BindingSecretJob(parent)
        , m_mode(mode)
    {
        s_lastInstance = this;
    }

    void start(const QString&, const QStringList&) override
    {
        Q_EMIT started();
        if (m_mode == AutoSuccess) {
            QTimer::singleShot(0, this, [this]() { emitSuccess(); });
        } else if (m_mode == AutoFailure) {
            QTimer::singleShot(0, this, [this]() { emitFailure(QStringLiteral("auto-failure")); });
        }
    }

    void cancel() override
    {
        QTimer::singleShot(0, this, [this]() { emitFailure(QStringLiteral("cancelled")); });
    }

    void emitSuccess(const QByteArray& stdoutPayload = QByteArrayLiteral("{\"status\":\"ok\"}"))
    {
        Q_EMIT completed(true, QString(), stdoutPayload, QByteArray());
    }

    void emitFailure(const QString& message, const QByteArray& stdoutPayload = QByteArray())
    {
        Q_EMIT completed(false, message, stdoutPayload, QByteArray());
    }

    static MockBindingSecretJob* takeLastInstance()
    {
        auto* instance = s_lastInstance;
        s_lastInstance = nullptr;
        return instance;
    }

private:
    Mode m_mode;
    static MockBindingSecretJob* s_lastInstance;
};

MockBindingSecretJob* MockBindingSecretJob::s_lastInstance = nullptr;

} // namespace

class LicenseActivationControllerTest : public QObject {
    Q_OBJECT

private slots:
    void activatesWithValidLicense();
    void rejectsMismatchedFingerprint();
    void acceptsBase64Payload();
    void cancelsBindingSecretPriming();
    void propagatesBindingSecretFailureStatus();
    void autoProvisionImportsMatchingLicense();
    void autoProvisionRunsDuringInitialize();
    void updatesFingerprintWhenDocumentCreated();
    void tracksExternalLicenseChanges();
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
    controller.setBindingSecretJobFactory([](QObject* owner) {
        return new MockBindingSecretJob(MockBindingSecretJob::AutoSuccess, owner);
    });
    controller.setConfigDirectory(configDir);
    controller.setLicenseStoragePath(QDir(licenseDir).filePath(QStringLiteral("license.json")));
    controller.initialize();

    QSignalSpy primingSpy(&controller, &LicenseActivationController::bindingSecretPrimingFinished);
    QVERIFY(controller.loadLicenseFile(licenseFile));
    QVERIFY(primingSpy.wait());
    QCOMPARE(primingSpy.takeFirst().at(0).toBool(), true);
    QVERIFY(controller.licenseActive());
    QCOMPARE(controller.licenseFingerprint(), expectedFingerprint);
    QCOMPARE(controller.licenseEdition(), QStringLiteral("pro"));
    QCOMPARE(controller.licenseLicenseId(), QStringLiteral("demo-pro"));
    QCOMPARE(controller.licenseMaintenanceUntil(), QStringLiteral("2099-12-31"));
    QVERIFY(controller.licenseMaintenanceActive());
    QCOMPARE(controller.licenseModules(), QStringList({QStringLiteral("futures")}));
    QCOMPARE(controller.licenseRuntime(), QStringList({QStringLiteral("auto_trader")}));
    QCOMPARE(controller.licenseHolderName(), QStringLiteral("OEM-Control"));

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
    controller.setBindingSecretJobFactory([](QObject* owner) {
        return new MockBindingSecretJob(MockBindingSecretJob::Manual, owner);
    });
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
    controller.setBindingSecretJobFactory([](QObject* owner) {
        return new MockBindingSecretJob(MockBindingSecretJob::AutoSuccess, owner);
    });
    controller.setConfigDirectory(configDir);
    controller.setLicenseStoragePath(tempDir.filePath(QStringLiteral("var/licenses/active/license.json")));
    controller.initialize();

    const QByteArray encoded = buildLicenseDocument(fingerprint)
                                   .toJson(QJsonDocument::Compact)
                                   .toBase64();

    QSignalSpy primingSpy(&controller, &LicenseActivationController::bindingSecretPrimingFinished);
    QVERIFY(controller.applyLicenseText(QString::fromUtf8(encoded)));
    QVERIFY(primingSpy.wait());
    QCOMPARE(primingSpy.takeFirst().at(0).toBool(), true);
    QVERIFY(controller.licenseActive());
    QCOMPARE(controller.licenseFingerprint(), fingerprint);
    QCOMPARE(controller.licenseEdition(), QStringLiteral("pro"));
    QVERIFY(controller.statusMessage().contains(QStringLiteral("Licencja aktywna")));
}

void LicenseActivationControllerTest::cancelsBindingSecretPriming()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString configDir = tempDir.filePath(QStringLiteral("config"));
    QDir().mkpath(configDir);
    const QString fingerprint = QStringLiteral("DEVICE-CANCEL");
    writeFingerprintDocument(configDir, fingerprint);

    LicenseActivationController controller;
    controller.setBindingSecretJobFactory([](QObject* owner) {
        return new MockBindingSecretJob(MockBindingSecretJob::Manual, owner);
    });
    controller.setConfigDirectory(configDir);
    controller.setLicenseStoragePath(tempDir.filePath(QStringLiteral("var/licenses/active/license.json")));
    controller.initialize();

    const QByteArray encoded = buildLicenseDocument(fingerprint)
                                   .toJson(QJsonDocument::Compact)
                                   .toBase64();

    QSignalSpy primingSpy(&controller, &LicenseActivationController::bindingSecretPrimingFinished);
    QVERIFY(controller.applyLicenseText(QString::fromUtf8(encoded)));
    MockBindingSecretJob* job = MockBindingSecretJob::takeLastInstance();
    QVERIFY(job != nullptr);

    controller.cancelBindingSecretPriming();
    QVERIFY(primingSpy.wait());
    const QList<QVariant> arguments = primingSpy.takeFirst();
    QCOMPARE(arguments.at(0).toBool(), false);
    QVERIFY(controller.statusIsError());
    QVERIFY(controller.statusMessage().contains(QStringLiteral("anul"), Qt::CaseInsensitive));
    QFile persisted(controller.licenseStoragePath());
    QVERIFY(!persisted.exists());
    QVERIFY(!controller.licenseActive());
}

void LicenseActivationControllerTest::propagatesBindingSecretFailureStatus()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString configDir = tempDir.filePath(QStringLiteral("config"));
    QDir().mkpath(configDir);
    const QString fingerprint = QStringLiteral("DEVICE-FAILURE");
    writeFingerprintDocument(configDir, fingerprint);

    LicenseActivationController controller;
    controller.setBindingSecretJobFactory([](QObject* owner) {
        return new MockBindingSecretJob(MockBindingSecretJob::Manual, owner);
    });
    controller.setConfigDirectory(configDir);
    controller.setLicenseStoragePath(tempDir.filePath(QStringLiteral("var/licenses/active/license.json")));
    controller.initialize();

    const QByteArray encoded = buildLicenseDocument(fingerprint)
                                   .toJson(QJsonDocument::Compact)
                                   .toBase64();

    QSignalSpy primingSpy(&controller, &LicenseActivationController::bindingSecretPrimingFinished);
    QVERIFY(controller.applyLicenseText(QString::fromUtf8(encoded)));
    MockBindingSecretJob* job = MockBindingSecretJob::takeLastInstance();
    QVERIFY(job != nullptr);

    job->emitSuccess(QByteArrayLiteral("{\"status\":\"error\",\"error\":\"denied\"}"));

    QVERIFY(primingSpy.wait());
    const QList<QVariant> arguments = primingSpy.takeFirst();
    QCOMPARE(arguments.at(0).toBool(), false);
    QVERIFY(controller.statusIsError());
    QVERIFY(controller.statusMessage().contains(QStringLiteral("denied"), Qt::CaseInsensitive));
    QFile persisted(controller.licenseStoragePath());
    QVERIFY(!persisted.exists());
    QVERIFY(!controller.licenseActive());
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
    controller.setBindingSecretJobFactory([](QObject* owner) {
        return new MockBindingSecretJob(MockBindingSecretJob::AutoSuccess, owner);
    });
    controller.setConfigDirectory(configDir);
    controller.setProvisioningDirectory(inboxDir);
    controller.setLicenseStoragePath(tempDir.filePath(QStringLiteral("var/licenses/active/license.json")));
    controller.initialize();

    QVariantMap fingerprintDoc{{QStringLiteral("fingerprint"), fingerprint}};
    QSignalSpy primingSpy(&controller, &LicenseActivationController::bindingSecretPrimingFinished);
    QVERIFY(controller.autoProvision(fingerprintDoc));
    QVERIFY(primingSpy.wait());
    QCOMPARE(primingSpy.takeFirst().at(0).toBool(), true);
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
    controller.setBindingSecretJobFactory([](QObject* owner) {
        return new MockBindingSecretJob(MockBindingSecretJob::AutoSuccess, owner);
    });
    controller.setConfigDirectory(configDir);
    controller.setProvisioningDirectory(inboxDir);
    controller.setLicenseStoragePath(tempDir.filePath(QStringLiteral("var/licenses/active/license.json")));
    QSignalSpy primingSpy(&controller, &LicenseActivationController::bindingSecretPrimingFinished);
    controller.initialize();

    QVERIFY(primingSpy.wait());
    QCOMPARE(primingSpy.takeFirst().at(0).toBool(), true);

    QVERIFY(controller.licenseActive());
    QCOMPARE(controller.licenseFingerprint(), fingerprint);
    QFile archivedLicense(licenseFile + QStringLiteral(".applied"));
    QVERIFY(archivedLicense.exists());
}

void LicenseActivationControllerTest::updatesFingerprintWhenDocumentCreated()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString configDir = tempDir.filePath(QStringLiteral("config"));
    QDir().mkpath(configDir);

    LicenseActivationController controller;
    controller.setConfigDirectory(configDir);
    controller.setLicenseStoragePath(tempDir.filePath(QStringLiteral("var/licenses/active/license.json")));
    controller.initialize();

    QVERIFY(!controller.expectedFingerprintAvailable());

    QSignalSpy spy(&controller, &LicenseActivationController::expectedFingerprintChanged);

    const QString fingerprint = QStringLiteral("WATCH-12345");
    writeFingerprintDocument(configDir, fingerprint);

    QTRY_VERIFY_WITH_TIMEOUT(spy.count() > 0, 1000);
    QCOMPARE(controller.expectedFingerprint(), fingerprint);

    const QString fingerprint2 = QStringLiteral("WATCH-ABCDE");
    writeFingerprintDocument(configDir, fingerprint2);

    QTRY_VERIFY_WITH_TIMEOUT(spy.count() > 1, 1000);
    QCOMPARE(controller.expectedFingerprint(), fingerprint2);
}

void LicenseActivationControllerTest::tracksExternalLicenseChanges()
{
    QTemporaryDir tempDir;
    QVERIFY(tempDir.isValid());

    const QString configDir = tempDir.filePath(QStringLiteral("config"));
    QDir().mkpath(configDir);
    const QString fingerprint = QStringLiteral("TRACK-777");
    writeFingerprintDocument(configDir, fingerprint);

    const QString licensePath = tempDir.filePath(QStringLiteral("var/licenses/active/license.json"));

    LicenseActivationController controller;
    controller.setConfigDirectory(configDir);
    controller.setLicenseStoragePath(licensePath);
    controller.initialize();

    QVERIFY(!controller.licenseActive());

    QSignalSpy activeSpy(&controller, &LicenseActivationController::licenseActiveChanged);

    writeLicenseFile(licensePath, buildLicenseDocument(fingerprint));

    QTRY_VERIFY_WITH_TIMEOUT(activeSpy.count() > 0, 1000);
    QVERIFY(controller.licenseActive());
    QCOMPARE(controller.licenseFingerprint(), fingerprint);

    QFile::remove(licensePath);

    QTRY_VERIFY_WITH_TIMEOUT(activeSpy.count() > 1, 1000);
    QVERIFY(!controller.licenseActive());
    QVERIFY(controller.licenseFingerprint().isEmpty());
}

QTEST_MAIN(LicenseActivationControllerTest)
#include "LicenseActivationControllerTest.moc"
