#include "SecurityAdminController.hpp"

#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QIODevice>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLoggingCategory>
#include <QProcess>
#include <QtGlobal>

Q_LOGGING_CATEGORY(lcSecurityAdmin, "bot.shell.security.admin")

namespace {

QString expandPath(const QString& path)
{
    if (path.startsWith(QStringLiteral("~/"))) {
        return QDir::homePath() + path.mid(1);
    }
    if (path == QStringLiteral("~")) {
        return QDir::homePath();
    }
    return path;
}

} // namespace

SecurityAdminController::SecurityAdminController(QObject* parent)
    : QObject(parent)
{
}

void SecurityAdminController::setPythonExecutable(const QString& executable)
{
    if (executable.trimmed().isEmpty()) {
        return;
    }
    m_pythonExecutable = executable;
}

void SecurityAdminController::setProfilesPath(const QString& path)
{
    m_profilesPath = expandPath(path);
}

void SecurityAdminController::setLicensePath(const QString& path)
{
    m_licensePath = expandPath(path);
}

void SecurityAdminController::setLogPath(const QString& path)
{
    m_logPath = expandPath(path);
}

bool SecurityAdminController::refresh()
{
    if (m_busy) {
        return false;
    }
    m_busy = true;
    Q_EMIT busyChanged();

    const QByteArray overridePath = qgetenv("BOT_CORE_UI_SECURITY_STATE_PATH");
    bool ok = false;
    if (!overridePath.isEmpty()) {
        ok = loadStateFromFile(QString::fromUtf8(overridePath));
    } else {
        QStringList args;
        args << QStringLiteral("-m")
             << QStringLiteral("bot_core.security.ui_bridge")
             << QStringLiteral("dump");
        if (!m_licensePath.isEmpty()) {
            args << QStringLiteral("--license-path") << m_licensePath;
        }
        if (!m_profilesPath.isEmpty()) {
            args << QStringLiteral("--profiles-path") << m_profilesPath;
        }

        QByteArray stdoutData;
        QByteArray stderrData;
        ok = runBridge(args, &stdoutData, &stderrData) && loadStateFromJson(stdoutData);
        if (!stderrData.isEmpty()) {
            qCWarning(lcSecurityAdmin) << "Bridge stderr:" << QString::fromUtf8(stderrData);
        }
    }

    m_busy = false;
    Q_EMIT busyChanged();
    return ok;
}

bool SecurityAdminController::assignProfile(const QString& userId,
                                            const QStringList& roles,
                                            const QString& displayName)
{
    if (m_busy) {
        return false;
    }
    QString trimmedId = userId.trimmed();
    if (trimmedId.isEmpty()) {
        qCWarning(lcSecurityAdmin) << "Odmowa aktualizacji profilu – pusty identyfikator użytkownika";
        return false;
    }

    m_busy = true;
    Q_EMIT busyChanged();

    QStringList args;
    args << QStringLiteral("-m")
         << QStringLiteral("bot_core.security.ui_bridge")
         << QStringLiteral("assign-profile")
         << QStringLiteral("--user") << trimmedId;
    for (const QString& role : roles) {
        const QString trimmedRole = role.trimmed();
        if (!trimmedRole.isEmpty()) {
            args << QStringLiteral("--role") << trimmedRole;
        }
    }
    if (!displayName.trimmed().isEmpty()) {
        args << QStringLiteral("--display-name") << displayName.trimmed();
    }
    if (!m_profilesPath.isEmpty()) {
        args << QStringLiteral("--profiles-path") << m_profilesPath;
    }
    if (!m_logPath.isEmpty()) {
        args << QStringLiteral("--log-path") << m_logPath;
    }
    const QByteArray actor = qgetenv("BOT_CORE_UI_ADMIN_ACTOR");
    if (!actor.isEmpty()) {
        args << QStringLiteral("--actor") << QString::fromUtf8(actor);
    }

    QByteArray stdoutData;
    QByteArray stderrData;
    const bool ok = runBridge(args, &stdoutData, &stderrData);
    if (!stderrData.isEmpty()) {
        qCWarning(lcSecurityAdmin) << "Bridge stderr:" << QString::fromUtf8(stderrData);
    }
    bool shouldRefresh = false;
    if (ok) {
        QJsonParseError parseError{};
        const QJsonDocument doc = QJsonDocument::fromJson(stdoutData, &parseError);
        if (parseError.error != QJsonParseError::NoError) {
            qCWarning(lcSecurityAdmin) << "Niepoprawna odpowiedź bridge assign-profile" << parseError.errorString();
        } else if (doc.isObject()) {
            const QJsonObject obj = doc.object();
            const QString status = obj.value(QStringLiteral("status")).toString();
            if (status.compare(QStringLiteral("ok"), Qt::CaseInsensitive) == 0) {
                const QString message = QStringLiteral("Zaktualizowano profil %1").arg(trimmedId);
                Q_EMIT adminEventLogged(message);
                shouldRefresh = true;
            } else {
                qCWarning(lcSecurityAdmin) << "Bridge zwrócił status" << status;
            }
        }
    }

    m_busy = false;
    Q_EMIT busyChanged();

    if (shouldRefresh) {
        refresh();
    }

    return ok;
}

bool SecurityAdminController::removeProfile(const QString& userId)
{
    if (m_busy) {
        return false;
    }

    const QString trimmedId = userId.trimmed();
    if (trimmedId.isEmpty()) {
        qCWarning(lcSecurityAdmin) << "Odmowa usunięcia profilu – pusty identyfikator użytkownika";
        return false;
    }

    m_busy = true;
    Q_EMIT busyChanged();

    QStringList args;
    args << QStringLiteral("-m")
         << QStringLiteral("bot_core.security.ui_bridge")
         << QStringLiteral("remove-profile")
         << QStringLiteral("--user") << trimmedId;
    if (!m_profilesPath.isEmpty()) {
        args << QStringLiteral("--profiles-path") << m_profilesPath;
    }
    if (!m_logPath.isEmpty()) {
        args << QStringLiteral("--log-path") << m_logPath;
    }
    const QByteArray actor = qgetenv("BOT_CORE_UI_ADMIN_ACTOR");
    if (!actor.isEmpty()) {
        args << QStringLiteral("--actor") << QString::fromUtf8(actor);
    }

    QByteArray stdoutData;
    QByteArray stderrData;
    bool success = runBridge(args, &stdoutData, &stderrData);
    if (!stderrData.isEmpty()) {
        qCWarning(lcSecurityAdmin) << "Bridge stderr:" << QString::fromUtf8(stderrData);
    }

    bool shouldRefresh = false;
    if (success) {
        QJsonParseError parseError{};
        const QJsonDocument doc = QJsonDocument::fromJson(stdoutData, &parseError);
        if (parseError.error != QJsonParseError::NoError) {
            qCWarning(lcSecurityAdmin) << "Niepoprawna odpowiedź bridge remove-profile" << parseError.errorString();
            success = false;
        } else if (doc.isObject()) {
            const QJsonObject obj = doc.object();
            const QString status = obj.value(QStringLiteral("status")).toString();
            if (status.compare(QStringLiteral("ok"), Qt::CaseInsensitive) == 0) {
                const QString message = QStringLiteral("Usunięto profil %1").arg(trimmedId);
                Q_EMIT adminEventLogged(message);
                shouldRefresh = true;
            } else {
                qCWarning(lcSecurityAdmin) << "Bridge remove-profile zwrócił status" << status;
                success = false;
            }
        } else {
            success = false;
        }
    }

    m_busy = false;
    Q_EMIT busyChanged();

    if (shouldRefresh) {
        refresh();
    }

    return success;
}

bool SecurityAdminController::runBridge(const QStringList& arguments,
                                        QByteArray* stdoutData,
                                        QByteArray* stderrData) const
{
    QProcess process;
    process.setProgram(m_pythonExecutable);
    process.setArguments(arguments);
    process.start();
    if (!process.waitForFinished()) {
        qCWarning(lcSecurityAdmin) << "Nie udało się uruchomić bridge" << m_pythonExecutable
                                   << process.errorString();
        return false;
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        qCWarning(lcSecurityAdmin) << "Bridge zakończył się kodem" << process.exitCode();
        if (stderrData) {
            *stderrData = process.readAllStandardError();
        }
        if (stdoutData) {
            *stdoutData = process.readAllStandardOutput();
        }
        return false;
    }
    if (stdoutData) {
        *stdoutData = process.readAllStandardOutput();
    }
    if (stderrData) {
        *stderrData = process.readAllStandardError();
    }
    return true;
}

bool SecurityAdminController::loadStateFromJson(const QByteArray& data)
{
    QJsonParseError parseError{};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        qCWarning(lcSecurityAdmin) << "Nie udało się sparsować JSON bridge:" << parseError.errorString();
        return false;
    }
    const QJsonObject root = doc.object();
    const QJsonObject licenseObject = root.value(QStringLiteral("license")).toObject();
    QVariantMap license;
    for (auto it = licenseObject.begin(); it != licenseObject.end(); ++it) {
        license.insert(it.key(), it.value().toVariant());
    }
    m_licenseInfo = license;
    Q_EMIT licenseInfoChanged();

    QVariantList profiles;
    const QJsonArray profilesArray = root.value(QStringLiteral("profiles")).toArray();
    profiles.reserve(profilesArray.size());
    for (const QJsonValue& value : profilesArray) {
        if (!value.isObject()) {
            continue;
        }
        const QJsonObject obj = value.toObject();
        QVariantMap map;
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            map.insert(it.key(), it.value().toVariant());
        }
        profiles.append(map);
    }
    m_userProfiles = profiles;
    Q_EMIT userProfilesChanged();
    return true;
}

bool SecurityAdminController::loadStateFromFile(const QString& path)
{
    QFile file(path);
    if (!file.exists()) {
        qCWarning(lcSecurityAdmin) << "Plik zastępczego stanu bezpieczeństwa nie istnieje:" << path;
        return false;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCWarning(lcSecurityAdmin) << "Nie można otworzyć pliku" << path << file.errorString();
        return false;
    }
    const QByteArray data = file.readAll();
    return loadStateFromJson(data);
}

