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

public:
    int autoProvisionCalls = 0;
    QString lastProvisionFingerprint;
    QStringList savedFingerprints;
    QStringList appliedPayloads;

private:
    bool m_licenseActive = false;
    bool m_provisioningInProgress = false;
};

class FirstRunWizardE2ETest : public QObject {
    Q_OBJECT

private slots:
    void autoProvisionTriggeredOnVisible();
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

QTEST_MAIN(FirstRunWizardE2ETest)
#include "FirstRunWizardE2ETest.moc"
