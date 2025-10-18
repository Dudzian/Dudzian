#include "reporting/ReportCenterController.hpp"

#include <QDir>
#include <QFileInfo>
#include <QSet>
#include <algorithm>

namespace {

QString normalizedPath(const QString& path)
{
    return QDir::cleanPath(QDir(path).absolutePath());
}

} // namespace

ReportCenterController::ReportCenterController(QObject* parent)
    : QObject(parent)
{
    m_refreshTimer.setSingleShot(true);
    m_refreshTimer.setInterval(100);

    connect(&m_refreshTimer, &QTimer::timeout, this, &ReportCenterController::refreshExports);
    connect(&m_watcher, &QFileSystemWatcher::directoryChanged, this,
            [this](const QString&) { scheduleWatcherRefresh(); });
    connect(&m_watcher, &QFileSystemWatcher::fileChanged, this,
            [this](const QString&) { scheduleWatcherRefresh(); });
}

QString ReportCenterController::reportsRoot() const
{
    return m_reportsRoot;
}

void ReportCenterController::setReportsRoot(const QString& path)
{
    const QString normalized = path.trimmed().isEmpty() ? QString() : normalizedPath(path);
    if (m_reportsRoot == normalized)
        return;

    m_reportsRoot = normalized;
    Q_EMIT reportsRootChanged();
    refreshExports();
}

QStringList ReportCenterController::exports() const
{
    return m_exports;
}

QStringList ReportCenterController::watchedDirectories() const
{
    QStringList directories = m_watcher.directories();
    std::sort(directories.begin(), directories.end());
    return directories;
}

void ReportCenterController::scheduleWatcherRefresh()
{
    if (!m_refreshTimer.isActive())
        m_refreshTimer.start();
}

void ReportCenterController::refreshExports()
{
    const QStringList discoveredExports = collectExportFiles();
    if (discoveredExports != m_exports) {
        m_exports = discoveredExports;
        Q_EMIT exportsChanged();
    }

    rebuildWatcher();
}

void ReportCenterController::rebuildWatcher()
{
    const QStringList desiredDirectories = collectDirectoriesToWatch();

    const QStringList trackedFiles = m_watcher.files();
    if (!trackedFiles.isEmpty())
        m_watcher.removePaths(trackedFiles);

    if (desiredDirectories == m_lastWatchedDirectories) {
        return;
    }

    if (!m_lastWatchedDirectories.isEmpty())
        m_watcher.removePaths(m_lastWatchedDirectories);

    if (!desiredDirectories.isEmpty())
        m_watcher.addPaths(desiredDirectories);

    m_lastWatchedDirectories = desiredDirectories;
    Q_EMIT watcherRebuilt(m_lastWatchedDirectories);
}

QStringList ReportCenterController::collectExportFiles() const
{
    QStringList files;
    if (m_reportsRoot.isEmpty())
        return files;

    QDir rootDir(m_reportsRoot);
    if (!rootDir.exists())
        return files;

    QList<QDir> pending = { rootDir };
    QSet<QString> seenFiles;

    while (!pending.isEmpty()) {
        QDir current = pending.takeLast();
        const QFileInfoList subdirs = current.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable);
        for (const QFileInfo& subdirInfo : subdirs) {
            pending.append(QDir(subdirInfo.absoluteFilePath()));
        }

        const QFileInfoList fileInfos = current.entryInfoList(QDir::Files | QDir::Readable);
        for (const QFileInfo& info : fileInfos) {
            const QString absolutePath = normalizedPath(info.absoluteFilePath());
            if (seenFiles.contains(absolutePath))
                continue;
            seenFiles.insert(absolutePath);
            files.append(absolutePath);
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

QStringList ReportCenterController::collectDirectoriesToWatch() const
{
    QStringList directories;
    if (m_reportsRoot.isEmpty())
        return directories;

    QDir rootDir(m_reportsRoot);
    if (!rootDir.exists())
        return directories;

    QList<QDir> pending = { rootDir };
    QSet<QString> seenDirs;

    while (!pending.isEmpty()) {
        QDir current = pending.takeLast();
        const QString dirPath = normalizedPath(current.absolutePath());
        if (seenDirs.contains(dirPath))
            continue;
        seenDirs.insert(dirPath);
        directories.append(dirPath);

        const QFileInfoList subdirs = current.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable);
        for (const QFileInfo& subdirInfo : subdirs) {
            pending.append(QDir(subdirInfo.absoluteFilePath()));
        }
    }

    std::sort(directories.begin(), directories.end());
    return directories;
}

