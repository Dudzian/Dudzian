#include <QtTest/QtTest>

#include <QQmlComponent>
#include <QQmlContext>
#include <QQmlEngine>
#include <QScopedPointer>
#include <QStringList>
#include <QVariant>

class MockActivationController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariant fingerprint READ fingerprint NOTIFY fingerprintChanged)

public:
    explicit MockActivationController(QObject* parent = nullptr)
        : QObject(parent)
    {
    }

    QVariant fingerprint() const { return m_fingerprint; }

    void setFingerprint(const QVariantMap& map)
    {
        m_fingerprint = map;
        Q_EMIT fingerprintChanged();
    }

    Q_INVOKABLE void refresh()
    {
        ++refreshCount;
        Q_EMIT refreshed();
    }

Q_SIGNALS:
    void fingerprintChanged();
    void refreshed();

public:
    int refreshCount = 0;

private:
    QVariant m_fingerprint;
};

class MockLicenseController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool licenseActive READ licenseActive WRITE setLicenseActive NOTIFY licenseActiveChanged)
    Q_PROPERTY(bool provisioningInProgress READ provisioningInProgress WRITE setProvisioningInProgress NOTIFY provisioningInProgressChanged)
    Q_PROPERTY(QString licenseEdition READ licenseEdition NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseLicenseId READ licenseLicenseId NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseMaintenanceUntil READ licenseMaintenanceUntil NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseFingerprint READ licenseFingerprint NOTIFY licenseDataChanged)

public:
    explicit MockLicenseController(QObject* parent = nullptr)
        : QObject(parent)
    {
    }

    bool licenseActive() const { return m_licenseActive; }
    void setLicenseActive(bool active)
    {
        if (m_licenseActive == active)
            return;
        m_licenseActive = active;
        Q_EMIT licenseActiveChanged();
    }

    bool provisioningInProgress() const { return m_provisioningInProgress; }
    void setProvisioningInProgress(bool inProgress)
    {
        if (m_provisioningInProgress == inProgress)
            return;
        m_provisioningInProgress = inProgress;
        Q_EMIT provisioningInProgressChanged();
    }

    QString licenseEdition() const { return m_licenseEdition; }
    void setLicenseEdition(const QString& edition)
    {
        if (m_licenseEdition == edition)
            return;
        m_licenseEdition = edition;
        Q_EMIT licenseDataChanged();
    }

    QString licenseLicenseId() const { return m_licenseId; }
    void setLicenseLicenseId(const QString& id)
    {
        if (m_licenseId == id)
            return;
        m_licenseId = id;
        Q_EMIT licenseDataChanged();
    }

    QString licenseMaintenanceUntil() const { return m_licenseMaintenanceUntil; }
    void setLicenseMaintenanceUntil(const QString& value)
    {
        if (m_licenseMaintenanceUntil == value)
            return;
        m_licenseMaintenanceUntil = value;
        Q_EMIT licenseDataChanged();
    }

    QString licenseFingerprint() const { return m_licenseFingerprint; }
    void setLicenseFingerprint(const QString& fingerprint)
    {
        if (m_licenseFingerprint == fingerprint)
            return;
        m_licenseFingerprint = fingerprint;
        Q_EMIT licenseDataChanged();
    }

    Q_INVOKABLE bool autoProvision(const QVariant& fingerprintDocument)
    {
        ++autoProvisionCalls;
        if (fingerprintDocument.canConvert<QVariantMap>()) {
            const QVariantMap map = fingerprintDocument.toMap();
            lastProvisionFingerprint = map.value(QStringLiteral("fingerprint")).toString();
            if (lastProvisionFingerprint.isEmpty()) {
                const QVariant payloadVariant = map.value(QStringLiteral("payload"));
                if (payloadVariant.canConvert<QVariantMap>()) {
                    const QVariantMap payload = payloadVariant.toMap();
                    lastProvisionFingerprint = payload.value(QStringLiteral("fingerprint")).toString();
                }
            }
        }
        return true;
    }

    Q_INVOKABLE bool saveExpectedFingerprint(const QString& fingerprint)
    {
        savedFingerprints.append(fingerprint);
        return true;
    }

    Q_INVOKABLE bool applyLicenseText(const QString& payload)
    {
        appliedPayloads.append(payload);
        return true;
    }

Q_SIGNALS:
    void licenseActiveChanged();
    void provisioningInProgressChanged();
    void licenseDataChanged();

public:
    int autoProvisionCalls = 0;
    QString lastProvisionFingerprint;
    QStringList savedFingerprints;
    QStringList appliedPayloads;

private:
    bool m_licenseActive = false;
    bool m_provisioningInProgress = false;
    QString m_licenseEdition;
    QString m_licenseId;
    QString m_licenseMaintenanceUntil;
    QString m_licenseFingerprint;
};

class FirstRunWizardE2ETest : public QObject {
    Q_OBJECT

private slots:
    void autoProvisionTriggeredOnVisible();
    void storesAuditEntryInLocalDatabase();
    void prunesAuditHistoryToMaxRows();
};

void FirstRunWizardE2ETest::autoProvisionTriggeredOnVisible()
{
    QQmlEngine engine;
    MockActivationController activation;
    MockLicenseController license;

    QVariantMap fingerprintPayload{{QStringLiteral("fingerprint"), QStringLiteral("DEVICE-XYZ")}};
    QVariantMap fingerprintDoc{{QStringLiteral("payload"), fingerprintPayload}};
    activation.setFingerprint(fingerprintDoc);

    engine.rootContext()->setContextProperty(QStringLiteral("activationController"), &activation);
    engine.rootContext()->setContextProperty(QStringLiteral("licenseController"), &license);

    QQmlComponent component(&engine, QUrl(QStringLiteral("qrc:/qml/components/FirstRunWizard.qml")));
    QObject* object = component.create(engine.rootContext());
    QVERIFY2(object, qPrintable(component.errorString()));
    QScopedPointer<QObject> guard(object);

    QTRY_COMPARE_WITH_TIMEOUT(activation.refreshCount, 1, 1000);
    QTRY_COMPARE_WITH_TIMEOUT(license.autoProvisionCalls, 1, 1000);
    QCOMPARE(license.lastProvisionFingerprint, QStringLiteral("DEVICE-XYZ"));
}

void FirstRunWizardE2ETest::storesAuditEntryInLocalDatabase()
{
    QQmlEngine engine;
    MockActivationController activation;
    MockLicenseController license;

    QVariantMap fingerprintPayload{{QStringLiteral("fingerprint"), QStringLiteral("DEVICE-XYZ")}};
    QVariantMap fingerprintDoc{{QStringLiteral("payload"), fingerprintPayload}};
    activation.setFingerprint(fingerprintDoc);

    license.setLicenseActive(true);
    license.setLicenseEdition(QStringLiteral("Enterprise"));
    license.setLicenseLicenseId(QStringLiteral("OEM-123"));
    license.setLicenseMaintenanceUntil(QStringLiteral("2025-01-01T00:00:00Z"));
    license.setLicenseFingerprint(QStringLiteral("DEVICE-XYZ"));

    engine.rootContext()->setContextProperty(QStringLiteral("activationController"), &activation);
    engine.rootContext()->setContextProperty(QStringLiteral("licenseController"), &license);

    QQmlComponent component(&engine, QUrl(QStringLiteral("qrc:/qml/components/security/LicenseActivationView.qml")));
    QObject* object = component.create(engine.rootContext());
    QVERIFY2(object, qPrintable(component.errorString()));
    QScopedPointer<QObject> guard(object);

    QMetaObject::invokeMethod(object, "clearAudit");

    bool stored = false;
    QVERIFY(QMetaObject::invokeMethod(object, "storeActiveLicense", Q_RETURN_ARG(bool, stored)));
    QVERIFY(stored);

    QVariant auditCount;
    QVERIFY(QMetaObject::invokeMethod(object, "auditCount", Q_RETURN_ARG(QVariant, auditCount)));
    QVERIFY(auditCount.toInt() >= 1);
}

void FirstRunWizardE2ETest::prunesAuditHistoryToMaxRows()
{
    QQmlEngine engine;
    MockActivationController activation;
    MockLicenseController license;

    QVariantMap fingerprintPayload{{QStringLiteral("fingerprint"), QStringLiteral("DEVICE-XYZ")}};
    QVariantMap fingerprintDoc{{QStringLiteral("payload"), fingerprintPayload}};
    activation.setFingerprint(fingerprintDoc);

    license.setLicenseActive(true);
    license.setLicenseEdition(QStringLiteral("Enterprise"));
    license.setLicenseLicenseId(QStringLiteral("OEM-123"));
    license.setLicenseMaintenanceUntil(QStringLiteral("2025-01-01T00:00:00Z"));

    engine.rootContext()->setContextProperty(QStringLiteral("activationController"), &activation);
    engine.rootContext()->setContextProperty(QStringLiteral("licenseController"), &license);

    QQmlComponent component(&engine, QUrl(QStringLiteral("qrc:/qml/components/security/LicenseActivationView.qml")));
    QObject* object = component.create(engine.rootContext());
    QVERIFY2(object, qPrintable(component.errorString()));
    QScopedPointer<QObject> guard(object);

    QMetaObject::invokeMethod(object, "clearAudit");

    for (int i = 0; i < 210; ++i) {
        license.setLicenseFingerprint(QStringLiteral("DEVICE-%1").arg(i));
        bool stored = false;
        QVERIFY(QMetaObject::invokeMethod(object, "storeActiveLicense", Q_RETURN_ARG(bool, stored)));
        QVERIFY(stored);
    }

    QVariant auditCount;
    QVERIFY(QMetaObject::invokeMethod(object, "auditCount", Q_RETURN_ARG(QVariant, auditCount)));
    QCOMPARE(auditCount.toInt(), 200);
}

QTEST_MAIN(FirstRunWizardE2ETest)
#include "FirstRunWizardE2ETest.moc"
