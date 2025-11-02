#include "RuntimeUtils.hpp"

#include "PathUtils.hpp"

#include <QByteArray>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QObject>
#include <QProcess>

namespace bot::shell::utils {

namespace {

QString runtimeDirectory()
{
    const QByteArray overrideDir = qgetenv("BOT_CORE_UI_RUNTIME_DIR");
    if (!overrideDir.isEmpty())
        return expandPath(QString::fromUtf8(overrideDir));
    return expandPath(QStringLiteral("var/runtime"));
}

} // namespace

QString runtimeLockFilePath()
{
    const QByteArray overridePath = qgetenv("BOT_CORE_UI_LOCK_FILE");
    if (!overridePath.isEmpty())
        return expandPath(QString::fromUtf8(overridePath));

    const QString runtimeDir = runtimeDirectory();
    QDir dir(runtimeDir);
    return dir.filePath(QStringLiteral("bot_trading_shell.lock"));
}

bool ensureLockFileDirectory(const QString& lockPath, QString* errorMessage)
{
    const QFileInfo info(lockPath);
    QDir directory = info.dir();
    if (directory.exists())
        return true;
    if (directory.mkpath(QStringLiteral(".")))
        return true;
    if (errorMessage) {
        *errorMessage = QObject::tr("Nie udało się utworzyć katalogu blokady instancji (%1).")
                            .arg(directory.absolutePath());
    }
    return false;
}

QString detectSecurityPythonExecutable()
{
    const QStringList args = QCoreApplication::arguments();
    for (int i = 1; i < args.size(); ++i) {
        const QString& entry = args.at(i);
        if (entry.startsWith(QStringLiteral("--security-python="))) {
            const QString value = entry.section(QLatin1Char('='), 1);
            if (!value.trimmed().isEmpty())
                return expandPath(value.trimmed());
        }
        if (entry == QStringLiteral("--security-python")) {
            if (i + 1 < args.size()) {
                const QString value = args.at(i + 1).trimmed();
                if (!value.isEmpty())
                    return expandPath(value);
            }
        }
    }

    const QByteArray envOverride = qgetenv("BOT_CORE_UI_PYTHON");
    if (!envOverride.isEmpty())
        return expandPath(QString::fromUtf8(envOverride));

    return QStringLiteral("python3");
}

SingleInstanceGuard::SingleInstanceGuard(QString lockFilePath)
    : m_lockFile(std::move(lockFilePath))
{
    m_lockFile.setStaleLockTime(0);
}

SingleInstanceGuard::~SingleInstanceGuard() = default;

bool SingleInstanceGuard::tryAcquire(int timeoutMs)
{
    m_error.clear();
    m_conflict = {};
    m_locked = false;
    m_lastError = QLockFile::NoError;

    if (m_lockFile.tryLock(timeoutMs)) {
        m_locked = true;
        return true;
    }

    m_lastError = m_lockFile.error();

    if (m_lastError == QLockFile::LockFailedError) {
        if (m_lockFile.removeStaleLockFile() && m_lockFile.tryLock(timeoutMs)) {
            m_locked = true;
            m_lastError = QLockFile::NoError;
            return true;
        }
        m_error = QObject::tr("Aplikacja jest już uruchomiona.");
        m_lockFile.getLockInfo(&m_conflict.pid, &m_conflict.hostname, &m_conflict.applicationId);
        return false;
    }

    if (m_lastError == QLockFile::PermissionError) {
        m_error = QObject::tr("Brak uprawnień do utworzenia blokady instancji.");
    } else {
        m_error = QObject::tr("Nieoczekiwany błąd blokady instancji.");
    }
    return false;
}

bool SingleInstanceGuard::isHeld() const
{
    return m_locked;
}

QString SingleInstanceGuard::errorString() const
{
    return m_error;
}

LockConflictInfo SingleInstanceGuard::conflictInfo() const
{
    return m_conflict;
}

QString SingleInstanceGuard::lockFilePath() const
{
    return m_lockFile.fileName();
}

bool SingleInstanceGuard::hasConflict() const
{
    return m_lastError == QLockFile::LockFailedError;
}

QLockFile::LockError SingleInstanceGuard::lastError() const
{
    return m_lastError;
}

bool reportSingleInstanceConflict(const QString& pythonExecutable,
                                  const QString& lockPath,
                                  const LockConflictInfo& conflict,
                                  QString* errorMessage,
                                  int timeoutMs)
{
    QString program = pythonExecutable.trimmed();
    if (program.isEmpty())
        program = QStringLiteral("python3");

    QStringList args;
    args << QStringLiteral("-m") << QStringLiteral("bot_core.security.fingerprint")
         << QStringLiteral("report-single-instance") << QStringLiteral("--lock-path") << lockPath;

    if (conflict.pid > 0)
        args << QStringLiteral("--owner-pid") << QString::number(conflict.pid);
    if (!conflict.hostname.isEmpty())
        args << QStringLiteral("--owner-host") << conflict.hostname;
    if (!conflict.applicationId.isEmpty())
        args << QStringLiteral("--owner-application") << conflict.applicationId;

    QProcess process;
    process.start(program, args);

    const int effectiveTimeout = timeoutMs > 0 ? timeoutMs : -1;
    if (!process.waitForStarted(effectiveTimeout)) {
        if (errorMessage) {
            const QString startError = process.errorString().trimmed();
            if (!startError.isEmpty()) {
                *errorMessage = startError;
            } else if (process.error() == QProcess::FailedToStart) {
                *errorMessage = QObject::tr(
                    "Nie udało się uruchomić procesu raportowania konfliktu instancji (%1)."
                ).arg(program);
            } else {
                *errorMessage = QObject::tr("Proces raportowania konfliktu instancji nie wystartował.");
            }
        }
        process.kill();
        process.waitForFinished(1000);
        return false;
    }

    if (!process.waitForFinished(effectiveTimeout)) {
        process.kill();
        process.waitForFinished(1000);
        if (errorMessage) {
            *errorMessage = QObject::tr("Raportowanie konfliktu instancji przekroczyło limit czasu.");
        }
        return false;
    }

    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        if (errorMessage) {
            const QString stderrText = QString::fromUtf8(process.readAllStandardError()).trimmed();
            if (!stderrText.isEmpty())
                *errorMessage = stderrText;
            else
                *errorMessage = QObject::tr("Raportowanie konfliktu instancji zakończyło się błędem.");
        }
        return false;
    }

    return true;
}

} // namespace bot::shell::utils
