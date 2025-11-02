#pragma once

#include <QLockFile>
#include <QString>

namespace bot::shell::utils {

struct LockConflictInfo {
    qint64 pid = 0;
    QString hostname;
    QString applicationId;
};

QString runtimeLockFilePath();

bool ensureLockFileDirectory(const QString& lockPath, QString* errorMessage = nullptr);

QString detectSecurityPythonExecutable();

class SingleInstanceGuard {
public:
    explicit SingleInstanceGuard(QString lockFilePath);
    SingleInstanceGuard(const SingleInstanceGuard&) = delete;
    SingleInstanceGuard& operator=(const SingleInstanceGuard&) = delete;
    SingleInstanceGuard(SingleInstanceGuard&&) noexcept = default;
    SingleInstanceGuard& operator=(SingleInstanceGuard&&) noexcept = default;
    ~SingleInstanceGuard();

    bool tryAcquire(int timeoutMs = 0);
    bool isHeld() const;
    QString errorString() const;
    LockConflictInfo conflictInfo() const;
    QString lockFilePath() const;
    bool hasConflict() const;
    QLockFile::LockError lastError() const;

private:
    QLockFile m_lockFile;
    QString m_error;
    LockConflictInfo m_conflict;
    bool m_locked = false;
    QLockFile::LockError m_lastError = QLockFile::NoError;
};

bool reportSingleInstanceConflict(const QString& pythonExecutable,
                                  const QString& lockPath,
                                  const LockConflictInfo& conflict,
                                  QString* errorMessage = nullptr,
                                  int timeoutMs = 5000);

} // namespace bot::shell::utils
