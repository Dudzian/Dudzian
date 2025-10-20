#pragma once

#include <QObject>
#include <QFileSystemWatcher>
#include <QJsonDocument>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QUrl>
#include <QVariantMap>

class LicenseActivationController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool licenseActive READ licenseActive NOTIFY licenseActiveChanged)
    Q_PROPERTY(QString statusMessage READ statusMessage NOTIFY statusMessageChanged)
    Q_PROPERTY(bool statusIsError READ statusIsError NOTIFY statusMessageChanged)
    Q_PROPERTY(QString licenseFingerprint READ licenseFingerprint NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseProfile READ licenseProfile NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseIssuer READ licenseIssuer NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseBundleVersion READ licenseBundleVersion NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseIssuedAt READ licenseIssuedAt NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseExpiresAt READ licenseExpiresAt NOTIFY licenseDataChanged)
    Q_PROPERTY(QStringList licenseFeatures READ licenseFeatures NOTIFY licenseDataChanged)
    Q_PROPERTY(QString expectedFingerprint READ expectedFingerprint NOTIFY expectedFingerprintChanged)
    Q_PROPERTY(bool expectedFingerprintAvailable READ expectedFingerprintAvailable NOTIFY expectedFingerprintChanged)
    Q_PROPERTY(bool provisioningInProgress READ provisioningInProgress NOTIFY provisioningInProgressChanged)
    Q_PROPERTY(QString provisioningDirectory READ provisioningDirectory WRITE setProvisioningDirectory NOTIFY provisioningDirectoryChanged)

public:
    explicit LicenseActivationController(QObject* parent = nullptr);

    void setConfigDirectory(const QString& path);
    void setLicenseStoragePath(const QString& path);
    void setFingerprintDocumentPath(const QString& path);
    Q_INVOKABLE void setProvisioningDirectory(const QString& path);
    void initialize();

    Q_INVOKABLE bool loadLicenseUrl(const QUrl& url);
    Q_INVOKABLE bool loadLicenseFile(const QString& path);
    Q_INVOKABLE bool applyLicenseText(const QString& text);
    Q_INVOKABLE QString licenseStoragePath() const;
    Q_INVOKABLE bool saveExpectedFingerprint(const QString& fingerprint);
    Q_INVOKABLE void overrideExpectedFingerprint(const QString& fingerprint);
    Q_INVOKABLE bool autoProvision(const QVariantMap& fingerprintDocument = QVariantMap());

    bool licenseActive() const { return m_licenseActive; }
    QString statusMessage() const { return m_statusMessage; }
    bool statusIsError() const { return m_statusIsError; }

    QString licenseFingerprint() const { return m_licenseFingerprint; }
    QString licenseProfile() const { return m_licenseProfile; }
    QString licenseIssuer() const { return m_licenseIssuer; }
    QString licenseBundleVersion() const { return m_licenseBundleVersion; }
    QString licenseIssuedAt() const { return m_licenseIssuedAt; }
    QString licenseExpiresAt() const { return m_licenseExpiresAt; }
    QStringList licenseFeatures() const { return m_licenseFeatures; }

    QString expectedFingerprint() const { return m_expectedFingerprint; }
    bool expectedFingerprintAvailable() const { return !m_expectedFingerprint.isEmpty(); }
    bool provisioningInProgress() const { return m_provisioningInProgress; }
    QString provisioningDirectory() const { return m_provisioningDirectory; }

signals:
    void licenseActiveChanged();
    void statusMessageChanged();
    void licenseDataChanged();
    void expectedFingerprintChanged();
    void licensePersisted(const QString& path);
    void provisioningInProgressChanged();
    void provisioningDirectoryChanged();

private:
    struct LicenseInfo {
        QString fingerprint;
        QString issuer;
        QString profile;
        QString bundleVersion;
        QString issuedAtIso;
        QString expiresAtIso;
        QStringList features;
        QJsonDocument document;
    };

    bool ensureInitialized();
    QString resolveLicenseOutputPath() const;
    QString resolveFingerprintDocumentPath() const;
    QString resolveProvisioningDirectory() const;
    void refreshExpectedFingerprint();
    void loadPersistedLicense();
    bool activateFromDocument(const QJsonDocument& document, bool persist, const QString& sourceDescription);
    bool parseLicenseDocument(const QJsonDocument& document, LicenseInfo& info, QString& error) const;
    bool persistLicense(const QJsonDocument& document);
    void setStatusMessage(const QString& message, bool isError);
    bool persistExpectedFingerprint(const QString& fingerprint, QString* errorMessage);
    bool provisionFromDirectory(const QString& directory, const QString& expectedFingerprint);
    static QString fingerprintFromVariant(const QVariantMap& fingerprintDocument);
    bool tryProvisionFile(const QString& path, const QString& expectedFingerprint);
    bool activateIfMatching(const QJsonDocument& document, bool persist, const QString& sourceDescription,
                            const QString& expectedFingerprint);
    static QString expandPath(const QString& path);
    void setupProvisioningWatcher();
    void handleProvisioningDirectoryEvent(const QString& path);
    void scheduleProvisioningScan(int delayMs = 0, bool reportNotFound = false);
    void attemptAutomaticProvisioning(bool reportNotFound);
    bool runProvisioningScan(const QString& expectedFingerprint, bool reportNotFound);

    bool m_initialized = false;
    bool m_licenseActive = false;
    bool m_statusIsError = false;
    bool m_provisioningInProgress = false;

    QString m_statusMessage;
    QString m_licenseFingerprint;
    QString m_licenseProfile;
    QString m_licenseIssuer;
    QString m_licenseBundleVersion;
    QString m_licenseIssuedAt;
    QString m_licenseExpiresAt;
    QStringList m_licenseFeatures;
    QJsonDocument m_lastDocument;

    QString m_configDirectory;
    QString m_licenseOutputPath;
    QString m_fingerprintDocumentPath;
    QString m_expectedFingerprint;
    QString m_provisioningDirectory;

    QFileSystemWatcher m_provisioningWatcher;
    QTimer m_provisioningScanTimer;
    bool m_pendingProvisioningError = false;
};
