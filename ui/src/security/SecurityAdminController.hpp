#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVariantList>

class SecurityAdminController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList userProfiles READ userProfiles NOTIFY userProfilesChanged)
    Q_PROPERTY(QVariantMap licenseInfo READ licenseInfo NOTIFY licenseInfoChanged)
    Q_PROPERTY(bool busy READ isBusy NOTIFY busyChanged)

public:
    explicit SecurityAdminController(QObject* parent = nullptr);

    QVariantList userProfiles() const { return m_userProfiles; }
    QVariantMap licenseInfo() const { return m_licenseInfo; }
    bool isBusy() const { return m_busy; }

    Q_INVOKABLE bool refresh();
    Q_INVOKABLE bool assignProfile(const QString& userId,
                                   const QStringList& roles,
                                   const QString& displayName = {});
    Q_INVOKABLE bool removeProfile(const QString& userId);

    void setPythonExecutable(const QString& executable);
    void setProfilesPath(const QString& path);
    void setLicensePath(const QString& path);
    void setLogPath(const QString& path);

signals:
    void userProfilesChanged();
    void licenseInfoChanged();
    void busyChanged();
    void adminEventLogged(const QString& message);

private:
    bool runBridge(const QStringList& arguments, QByteArray* stdoutData, QByteArray* stderrData) const;
    bool loadStateFromJson(const QByteArray& data);
    bool loadStateFromFile(const QString& path);

    QString m_pythonExecutable = QStringLiteral("python3");
    QString m_profilesPath;
    QString m_licensePath;
    QString m_logPath;
    QVariantList m_userProfiles;
    QVariantMap m_licenseInfo;
    bool m_busy = false;
};

