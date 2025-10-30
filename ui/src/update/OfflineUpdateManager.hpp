#pragma once

#include <QObject>
#include <QPointer>
#include <QVariantList>
#include <QVariantMap>
#include <QStringList>

class LicenseActivationController;

class OfflineUpdateManager : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(QVariantList availableUpdates READ availableUpdates NOTIFY availableUpdatesChanged)
    Q_PROPERTY(QVariantList installedUpdates READ installedUpdates NOTIFY installedUpdatesChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY lastErrorChanged)

public:
    explicit OfflineUpdateManager(QObject* parent = nullptr);

    bool busy() const { return m_busy; }
    QVariantList availableUpdates() const { return m_availableUpdates; }
    QVariantList installedUpdates() const { return m_installedUpdates; }
    QString lastError() const { return m_lastError; }

    void setPackagesDirectory(const QString& path);
    void setInstallDirectory(const QString& path);
    void setStateFilePath(const QString& path);
    void setLicenseController(LicenseActivationController* controller);
    void setFingerprintOverride(const QString& fingerprint);
    void setTpmEvidencePath(const QString& path);

    Q_INVOKABLE bool refresh();
    Q_INVOKABLE bool applyUpdate(const QString& packageId);
    Q_INVOKABLE bool rollbackUpdate(const QString& packageId);
    Q_INVOKABLE bool applyDifferentialPatch(const QString& baseId, const QString& patchId);
    Q_INVOKABLE QVariantMap describeUpdate(const QString& packageId) const;

signals:
    void busyChanged();
    void availableUpdatesChanged();
    void installedUpdatesChanged();
    void lastErrorChanged();
    void updateProgress(const QString& packageId, double progress);
    void updateCompleted(const QString& packageId);
    void updateFailed(const QString& packageId, const QString& reason);

private:
    struct UpdatePackage {
        QString id;
        QString version;
        QString fingerprint;
        QString signature;
        QString path;
        bool differential = false;
        QString baseId;
        QVariantMap metadata;
        QString payloadFile;
        QString diffFile;
        QVariantMap signatureObject;
        QVariantMap integrity;
    };

    void setBusy(bool busy);
    bool verifyPackageSignature(const UpdatePackage& pkg, QString* message) const;
    bool verifyFingerprint(const UpdatePackage& pkg, QString* message) const;
    bool loadState();
    bool persistState() const;
    QList<UpdatePackage> loadPackages(QString* errorMessage) const;
    bool applyPackage(const UpdatePackage& pkg);
    bool copyPackagePayload(const UpdatePackage& pkg, QString* errorMessage);
    bool applyDifferential(const UpdatePackage& basePkg, const UpdatePackage& patchPkg);
    QString storedPayloadPath(const UpdatePackage& pkg) const;
    QString storedArchivePath(const UpdatePackage& pkg) const;
    QString packagePayloadPath(const UpdatePackage& pkg) const;
    bool ensureEmptyDirectory(const QString& path, QString* errorMessage) const;
    bool extractArchive(const QString& archivePath, const QString& targetDir, QString* errorMessage) const;
    bool createArchive(const QString& sourceDir, const QString& archivePath, QString* errorMessage) const;
    bool overlayDirectory(const QString& sourceDir, const QString& targetDir, QString* errorMessage) const;
    bool removePaths(const QStringList& relativePaths, const QString& rootDir, QString* errorMessage) const;
    QStringList installedIds() const;
    UpdatePackage packageById(const QString& id) const;

    QString m_packagesDir;
    QString m_installDir;
    QString m_stateFile;
    QString m_fingerprintOverride;
    QString m_tpmEvidencePath;
    QPointer<LicenseActivationController> m_licenseController;

    bool m_busy = false;
    QVariantList m_availableUpdates;
    QVariantList m_installedUpdates;
    QString m_lastError;
};
