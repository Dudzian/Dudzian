#pragma once

#include <QObject>
#include <QByteArray>
#include <QFileSystemWatcher>
#include <QJsonDocument>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QUrl>
#include <QVariantMap>
#include <functional>
#include <memory>

class LicenseActivationController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool licenseActive READ licenseActive NOTIFY licenseActiveChanged)
    Q_PROPERTY(QString statusMessage READ statusMessage NOTIFY statusMessageChanged)
    Q_PROPERTY(bool statusIsError READ statusIsError NOTIFY statusMessageChanged)
    Q_PROPERTY(QString licenseFingerprint READ licenseFingerprint NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseEdition READ licenseEdition NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseLicenseId READ licenseLicenseId NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseIssuedAt READ licenseIssuedAt NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseMaintenanceUntil READ licenseMaintenanceUntil NOTIFY licenseDataChanged)
    Q_PROPERTY(bool licenseMaintenanceActive READ licenseMaintenanceActive NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseHolderName READ licenseHolderName NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseHolderEmail READ licenseHolderEmail NOTIFY licenseDataChanged)
    Q_PROPERTY(int licenseSeats READ licenseSeats NOTIFY licenseDataChanged)
    Q_PROPERTY(bool licenseTrialActive READ licenseTrialActive NOTIFY licenseDataChanged)
    Q_PROPERTY(QString licenseTrialExpiresAt READ licenseTrialExpiresAt NOTIFY licenseDataChanged)
    Q_PROPERTY(QStringList licenseModules READ licenseModules NOTIFY licenseDataChanged)
    Q_PROPERTY(QStringList licenseEnvironments READ licenseEnvironments NOTIFY licenseDataChanged)
    Q_PROPERTY(QStringList licenseRuntime READ licenseRuntime NOTIFY licenseDataChanged)
    Q_PROPERTY(QString expectedFingerprint READ expectedFingerprint NOTIFY expectedFingerprintChanged)
    Q_PROPERTY(bool expectedFingerprintAvailable READ expectedFingerprintAvailable NOTIFY expectedFingerprintChanged)
    Q_PROPERTY(bool provisioningInProgress READ provisioningInProgress NOTIFY provisioningInProgressChanged)
    Q_PROPERTY(QString provisioningDirectory READ provisioningDirectory WRITE setProvisioningDirectory NOTIFY provisioningDirectoryChanged)

public:
    class BindingSecretJob : public QObject {
        Q_OBJECT

    public:
        explicit BindingSecretJob(QObject* parent = nullptr)
            : QObject(parent)
        {
        }

        virtual void start(const QString& program, const QStringList& arguments) = 0;
        virtual void cancel() = 0;

    signals:
        void started();
        void completed(bool success, const QString& message, const QByteArray& stdoutData, const QByteArray& stderrData);
    };

    explicit LicenseActivationController(QObject* parent = nullptr);

    void setConfigDirectory(const QString& path);
    void setLicenseStoragePath(const QString& path);
    void setFingerprintDocumentPath(const QString& path);
    Q_INVOKABLE void setProvisioningDirectory(const QString& path);
    Q_INVOKABLE void setPythonExecutable(const QString& executable);
    Q_INVOKABLE void setBindingSecretPath(const QString& path);
    void initialize();

    Q_INVOKABLE bool loadLicenseUrl(const QUrl& url);
    Q_INVOKABLE bool loadLicenseFile(const QString& path);
    Q_INVOKABLE bool applyLicenseText(const QString& text);
    Q_INVOKABLE QString licenseStoragePath() const;
    Q_INVOKABLE bool saveExpectedFingerprint(const QString& fingerprint);
    Q_INVOKABLE void overrideExpectedFingerprint(const QString& fingerprint);
    Q_INVOKABLE bool autoProvision(const QVariantMap& fingerprintDocument = QVariantMap());
    Q_INVOKABLE void cancelBindingSecretPriming();

    void setBindingSecretJobFactory(const std::function<BindingSecretJob*(QObject*)>& factory);
    BindingSecretJob* currentBindingSecretJob() const { return m_bindingSecretJob; }
    void setBindingSecretTimeout(int timeoutMs);

    bool licenseActive() const { return m_licenseActive; }
    QString statusMessage() const { return m_statusMessage; }
    bool statusIsError() const { return m_statusIsError; }

    QString licenseFingerprint() const { return m_licenseFingerprint; }
    QString licenseEdition() const { return m_licenseEdition; }
    QString licenseLicenseId() const { return m_licenseLicenseId; }
    QString licenseIssuedAt() const { return m_licenseIssuedAt; }
    QString licenseMaintenanceUntil() const { return m_licenseMaintenanceUntil; }
    bool licenseMaintenanceActive() const { return m_licenseMaintenanceActive; }
    QString licenseHolderName() const { return m_licenseHolderName; }
    QString licenseHolderEmail() const { return m_licenseHolderEmail; }
    int licenseSeats() const { return m_licenseSeats; }
    bool licenseTrialActive() const { return m_licenseTrialActive; }
    QString licenseTrialExpiresAt() const { return m_licenseTrialExpiresAt; }
    QStringList licenseModules() const { return m_licenseModules; }
    QStringList licenseEnvironments() const { return m_licenseEnvironments; }
    QStringList licenseRuntime() const { return m_licenseRuntime; }

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
    void bindingSecretPrimingStarted();
    void bindingSecretPrimingFinished(bool success, const QString& message);

private:
    struct LicenseInfo {
        QString fingerprint;
        QString licenseId;
        QString edition;
        QString issuedAtIso;
        QString maintenanceUntilIso;
        bool maintenanceActive = false;
        QString holderName;
        QString holderEmail;
        int seats = 0;
        bool trialActive = false;
        QString trialExpiresIso;
        QStringList modules;
        QStringList environments;
        QStringList runtime;
        QJsonDocument document;
    };

    bool ensureInitialized();
    QString resolveLicenseOutputPath() const;
    QString resolveFingerprintDocumentPath() const;
    QString resolveProvisioningDirectory() const;
    QString resolveBindingSecretPath() const;
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
    QString m_licenseEdition;
    QString m_licenseLicenseId;
    QString m_licenseIssuedAt;
    QString m_licenseMaintenanceUntil;
    bool m_licenseMaintenanceActive = false;
    QString m_licenseHolderName;
    QString m_licenseHolderEmail;
    int m_licenseSeats = 0;
    bool m_licenseTrialActive = false;
    QString m_licenseTrialExpiresAt;
    QStringList m_licenseModules;
    QStringList m_licenseEnvironments;
    QStringList m_licenseRuntime;
    QJsonDocument m_lastDocument;

    QString m_configDirectory;
    QString m_licenseOutputPath;
    QString m_fingerprintDocumentPath;
    QString m_expectedFingerprint;
    QString m_provisioningDirectory;
    QString m_pythonExecutable;
    QString m_bindingSecretPath;

    QFileSystemWatcher m_provisioningWatcher;
    QTimer m_provisioningScanTimer;
    bool m_pendingProvisioningError = false;

    QFileSystemWatcher m_licenseWatcher;
    QFileSystemWatcher m_fingerprintWatcher;
    QTimer m_licenseReloadTimer;
    QTimer m_fingerprintReloadTimer;

    std::function<BindingSecretJob*(QObject*)> m_bindingSecretJobFactory;
    BindingSecretJob* m_bindingSecretJob = nullptr;
    QTimer m_bindingSecretTimeoutTimer;
    int m_bindingSecretTimeoutMs = 30000;
    enum class BindingSecretAbortReason {
        None,
        Cancelled,
        TimedOut,
    };
    BindingSecretAbortReason m_bindingSecretAbortReason = BindingSecretAbortReason::None;
    QByteArray m_bindingSecretStdout;
    QByteArray m_bindingSecretStderr;
    struct PendingActivation {
        LicenseInfo info;
        QString sourceDescription;
        bool persist = false;
    };
    std::unique_ptr<PendingActivation> m_pendingActivation;

    void setupLicenseWatcher();
    void setupFingerprintWatcher();
    void handleLicensePathEvent(const QString& path);
    void handleFingerprintPathEvent(const QString& path);
    void scheduleLicenseReload(int delayMs = 0);
    void scheduleFingerprintReload(int delayMs = 0);
    void clearLicenseState();
    bool primeBindingSecret(const QString& fingerprint);
};
