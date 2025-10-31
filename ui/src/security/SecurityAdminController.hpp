#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVariantList>
#include <QVariantMap>

class SecurityAdminController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList userProfiles READ userProfiles NOTIFY userProfilesChanged)
    Q_PROPERTY(QVariantMap licenseInfo READ licenseInfo NOTIFY licenseInfoChanged)
    Q_PROPERTY(bool busy READ isBusy NOTIFY busyChanged)
    Q_PROPERTY(QVariantMap tpmStatus READ tpmStatus NOTIFY tpmStatusChanged)
    Q_PROPERTY(QVariantMap integrityReport READ integrityReport NOTIFY integrityReportChanged)
    Q_PROPERTY(QVariantList auditLog READ auditLog NOTIFY auditLogChanged)

public:
    explicit SecurityAdminController(QObject* parent = nullptr);

    QVariantList userProfiles() const { return m_userProfiles; }
    QVariantMap licenseInfo() const { return m_licenseInfo; }
    bool isBusy() const { return m_busy; }
    QVariantMap tpmStatus() const { return m_tpmStatus; }
    QVariantMap integrityReport() const { return m_lastIntegrityReport; }
    QVariantList auditLog() const { return m_auditLog; }

    Q_INVOKABLE bool refresh();
    Q_INVOKABLE bool assignProfile(const QString& userId,
                                   const QStringList& roles,
                                   const QString& displayName = {});
    Q_INVOKABLE bool removeProfile(const QString& userId);

    void setPythonExecutable(const QString& executable);
    void setProfilesPath(const QString& path);
    void setLicensePath(const QString& path);
    void setLogPath(const QString& path);
    void setAlertsPath(const QString& path);
    void setAdditionalLogPaths(const QStringList& paths);
    void setTpmQuotePath(const QString& path);
    void setTpmKeyringPath(const QString& path);
    void setIntegrityManifestPath(const QString& path);

    Q_INVOKABLE bool verifyTpmBinding();
    Q_INVOKABLE bool runIntegrityCheck();
    Q_INVOKABLE bool exportSignedAuditLog(const QString& destinationDir);
    Q_INVOKABLE void ingestSecurityEvent(const QString& category,
                                         const QString& message,
                                         const QVariantMap& details = {},
                                         int severity = 1);

signals:
    void userProfilesChanged();
    void licenseInfoChanged();
    void busyChanged();
    void adminEventLogged(const QString& message);
    void tpmStatusChanged();
    void integrityReportChanged();
    void auditLogChanged();
    void securityAlertRaised(const QString& id, int severity, const QString& title, const QString& message);

private:
    bool runBridge(const QStringList& arguments, QByteArray* stdoutData, QByteArray* stderrData) const;
    bool loadStateFromJson(const QByteArray& data);
    bool loadStateFromFile(const QString& path);
    void recordAuditEvent(const QString& category, const QString& message, const QVariantMap& details = {});
    QString computeFileDigest(const QString& path) const;
    bool evaluateIntegrityManifest(QVariantMap* report);

    QString m_pythonExecutable = QStringLiteral("python3");
    QString m_profilesPath;
    QString m_licensePath;
    QString m_logPath;
    QString m_alertsPath;
    QStringList m_additionalLogPaths;
    QString m_tpmQuotePath;
    QString m_tpmKeyringPath;
    QString m_integrityManifestPath;
    QVariantList m_userProfiles;
    QVariantMap m_licenseInfo;
    QVariantMap m_tpmStatus;
    QVariantMap m_lastIntegrityReport;
    QVariantList m_auditLog;
    bool m_busy = false;
};

