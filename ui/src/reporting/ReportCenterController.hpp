#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVariantList>

class ReportCenterController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList reports READ reports NOTIFY reportsChanged)
    Q_PROPERTY(bool busy READ isBusy NOTIFY busyChanged)

public:
    explicit ReportCenterController(QObject* parent = nullptr);

    QString reportsRoot() const;
    void setReportsRoot(const QString& path);

    QStringList exports() const;

    QStringList watchedDirectories() const;

public Q_SLOTS:
    void rebuildWatcher();
    void scheduleWatcherRefresh();

Q_SIGNALS:
    void reportsRootChanged();
    void exportsChanged();
    void watcherRebuilt(const QStringList& directories);

private Q_SLOTS:
    void refreshExports();

private:
    QStringList collectExportFiles() const;
    QStringList collectDirectoriesToWatch() const;

    QFileSystemWatcher m_watcher;
    QTimer m_refreshTimer;
    QString m_reportsRoot;
    QStringList m_exports;
    QStringList m_lastWatchedDirectories;
};

