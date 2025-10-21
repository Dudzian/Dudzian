#pragma once

#include <QObject>
#include <QDateTime>
#include <QFileSystemWatcher>
#include <QString>
#include <QStringList>
#include <QThreadPool>
#include <QTimer>
#include <QUrl>
#include <QVariantList>
#include <QVariantMap>
#include <functional>

class ReportCenterController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList reports READ reports NOTIFY reportsChanged)
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(QString lastError READ lastError NOTIFY lastErrorChanged)
    Q_PROPERTY(QString lastNotification READ lastNotification NOTIFY lastNotificationChanged)
    Q_PROPERTY(QVariantList categories READ categories NOTIFY categoriesChanged)
    Q_PROPERTY(QVariantMap overviewStats READ overviewStats NOTIFY overviewStatsChanged)
    Q_PROPERTY(QVariantMap overviewPagination READ overviewPagination NOTIFY overviewPaginationChanged)
    Q_PROPERTY(QString archiveFormat READ archiveFormat WRITE setArchiveFormat NOTIFY archiveFormatChanged)
    Q_PROPERTY(int recentDaysFilter READ recentDaysFilter WRITE setRecentDaysFilter NOTIFY recentDaysFilterChanged)
    Q_PROPERTY(QDateTime sinceFilter READ sinceFilter WRITE setSinceFilter NOTIFY sinceFilterChanged)
    Q_PROPERTY(QDateTime untilFilter READ untilFilter WRITE setUntilFilter NOTIFY untilFilterChanged)
    Q_PROPERTY(QString categoryFilter READ categoryFilter WRITE setCategoryFilter NOTIFY categoryFilterChanged)
    Q_PROPERTY(QString summaryStatusFilter READ summaryStatusFilter WRITE setSummaryStatusFilter NOTIFY summaryStatusFilterChanged)
    Q_PROPERTY(QString exportsFilter READ exportsFilter WRITE setExportsFilter NOTIFY exportsFilterChanged)
    Q_PROPERTY(QString searchQuery READ searchQuery WRITE setSearchQuery NOTIFY searchQueryChanged)
    Q_PROPERTY(int limit READ limit WRITE setLimit NOTIFY limitChanged)
    Q_PROPERTY(int offset READ offset WRITE setOffset NOTIFY offsetChanged)
    Q_PROPERTY(QString sortKey READ sortKey WRITE setSortKey NOTIFY sortKeyChanged)
    Q_PROPERTY(QString sortDirection READ sortDirection WRITE setSortDirection NOTIFY sortDirectionChanged)
    Q_PROPERTY(QVariantList equityCurve READ equityCurve NOTIFY equityCurveChanged)
    Q_PROPERTY(QVariantList assetHeatmap READ assetHeatmap NOTIFY assetHeatmapChanged)

public:
    explicit ReportCenterController(QObject* parent = nullptr);

    QVariantList reports() const { return m_reports; }
    bool busy() const { return m_busy; }
    QString lastError() const { return m_lastError; }
    QString lastNotification() const { return m_lastNotification; }
    QVariantList categories() const { return m_categories; }
    QVariantMap overviewStats() const { return m_overviewStats; }
    QVariantMap overviewPagination() const { return m_overviewPagination; }
    QString archiveFormat() const { return m_archiveFormat; }
    int recentDaysFilter() const { return m_recentDaysFilter; }
    QDateTime sinceFilter() const { return m_sinceFilter; }
    QDateTime untilFilter() const { return m_untilFilter; }
    QString categoryFilter() const { return m_categoryFilter; }
    QString summaryStatusFilter() const { return m_summaryStatusFilter; }
    QString exportsFilter() const { return m_exportsFilter; }
    QString searchQuery() const { return m_searchQuery; }
    int limit() const { return m_limit; }
    int offset() const { return m_offset; }
    QString sortKey() const { return m_sortKey; }
    QString sortDirection() const { return m_sortDirection; }
    QVariantList equityCurve() const { return m_equityCurve; }
    QVariantList assetHeatmap() const { return m_assetHeatmap; }

    Q_INVOKABLE bool refresh();
    Q_INVOKABLE QVariantMap findReport(const QString& relativePath) const;
    Q_INVOKABLE bool saveReportAs(const QString& relativePath, const QUrl& destinationUrl);
    Q_INVOKABLE bool revealReport(const QString& relativePath);
    Q_INVOKABLE bool openExport(const QString& relativePath);
    Q_INVOKABLE bool deleteReport(const QString& relativePath);
    Q_INVOKABLE bool previewDeleteReport(const QString& relativePath);
    Q_INVOKABLE bool previewPurgeReports();
    Q_INVOKABLE bool purgeReports();
    Q_INVOKABLE bool previewArchiveReports(const QString& destination = QString(), bool overwrite = false, const QString& format = QString());
    Q_INVOKABLE bool archiveReports(const QString& destination = QString(), bool overwrite = false, const QString& format = QString());
    Q_INVOKABLE QString defaultArchiveDestination() const;

    void setPythonExecutable(const QString& executable);
    void setReportsDirectory(const QString& path);
    void setArchiveFormat(const QString& format);
    void setRecentDaysFilter(int days);
    void setSinceFilter(const QDateTime& since);
    void setUntilFilter(const QDateTime& until);
    void setCategoryFilter(const QString& category);
    void setSummaryStatusFilter(const QString& summaryStatus);
    void setExportsFilter(const QString& hasExports);
    void setSearchQuery(const QString& query);
    void setLimit(int limit);
    void setOffset(int offset);
    void setSortKey(const QString& sortKey);
    void setSortDirection(const QString& sortDirection);
    Q_INVOKABLE void clearSinceFilter();
    Q_INVOKABLE void clearUntilFilter();

signals:
    void reportsChanged();
    void busyChanged();
    void lastErrorChanged();
    void lastNotificationChanged();
    void categoriesChanged();
    void overviewStatsChanged();
    void overviewPaginationChanged();
    void recentDaysFilterChanged();
    void sinceFilterChanged();
    void untilFilterChanged();
    void categoryFilterChanged();
    void summaryStatusFilterChanged();
    void exportsFilterChanged();
    void searchQueryChanged();
    void limitChanged();
    void offsetChanged();
    void sortKeyChanged();
    void sortDirectionChanged();
    void archiveFormatChanged();
    void overviewReady(bool success);
    void deletePreviewReady(const QString& relativePath, const QVariantMap& result);
    void purgePreviewReady(const QVariantMap& result);
    void archivePreviewReady(const QString& destination, bool overwrite, const QString& format, const QVariantMap& result);
    void deleteFinished(const QString& relativePath, bool success);
    void purgeFinished(bool success);
    void archiveFinished(bool success);
    void equityCurveChanged();
    void assetHeatmapChanged();

private:
    struct BridgeResult {
        bool success = false;
        QByteArray stdoutData;
        QByteArray stderrData;
        QString errorMessage;
    };

    using BridgeCallback = std::function<void(const BridgeResult&)>;

    void runBridge(const QStringList& arguments, BridgeCallback&& callback);
    BridgeResult executeBridge(const QStringList& arguments) const;
    void beginTask();
    void endTask();
    bool loadOverview(const QByteArray& data);
    QString resolveReportsDirectory() const;
    static QString expandPath(const QString& path);
    void appendFilterArguments(QStringList& args, bool includePagination = true) const;
    void rebuildWatcher(const QString& baseDirectory, const QVariantList& reports);
    void scheduleWatcherRefresh();
    void handleWatcherTriggered();
    void setLastErrorMessage(const QString& message);
    void clearLastErrorMessage();
    void setLastNotificationMessage(const QString& message);
    void clearLastNotificationMessage();

    QString m_pythonExecutable = QStringLiteral("python3");
    QString m_reportsDirectory;
    QVariantList m_reports;
    bool m_busy = false;
    QString m_lastError;
    QString m_lastNotification;
    QVariantList m_categories;
    QVariantMap m_overviewStats;
    QVariantMap m_overviewPagination;
    QString m_archiveFormat = QStringLiteral("directory");
    QFileSystemWatcher m_watcher;
    QTimer m_watcherDebounce;
    QThreadPool m_workerPool;
    int m_recentDaysFilter = 0;
    QDateTime m_sinceFilter;
    QDateTime m_untilFilter;
    QString m_categoryFilter;
    QString m_summaryStatusFilter = QStringLiteral("any");
    QString m_exportsFilter = QStringLiteral("any");
    QString m_searchQuery;
    int m_limit = 0;
    int m_offset = 0;
    QString m_sortKey = QStringLiteral("updated_at");
    QString m_sortDirection = QStringLiteral("desc");
    int m_pendingTasks = 0;
    QVariantList m_equityCurve;
    QVariantList m_assetHeatmap;
};
