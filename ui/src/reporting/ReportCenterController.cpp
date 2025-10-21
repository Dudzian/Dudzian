#include "reporting/ReportCenterController.hpp"

#include <algorithm>

#include <QByteArray>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QFutureWatcher>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QLocale>
#include <QProcess>
#include <QSet>
#include <QSignalBlocker>
#include <QTimer>
#include <QUrl>
#include <QtGlobal>
#include <QtConcurrent>
#include <memory>

#include "utils/PathUtils.hpp"

Q_LOGGING_CATEGORY(lcReportCenter, "bot.shell.reporting")

namespace {
constexpr int kWatcherSoftLimit = 512;

QString cleanPath(const QString& path)
{
    if (path.trimmed().isEmpty())
        return {};
    QFileInfo info(path);
    if (info.isRelative())
        info.setFile(QDir::current().absoluteFilePath(path));
    return QDir::cleanPath(info.absoluteFilePath());
}

QString toFilePath(const QUrl& url)
{
    if (!url.isValid())
        return {};
    if (url.isLocalFile())
        return cleanPath(url.toLocalFile());
    return cleanPath(url.toString());
}

QString normalizeArchiveFormat(const QString& format)
{
    const QString normalized = format.trimmed().toLower();
    if (normalized == QLatin1String("zip") || normalized == QLatin1String("tar"))
        return normalized;
    return QStringLiteral("directory");
}

bool sameInstant(const QDateTime& lhs, const QDateTime& rhs)
{
    if (!lhs.isValid() && !rhs.isValid())
        return true;
    if (lhs.isValid() != rhs.isValid())
        return false;
    return lhs.toUTC() == rhs.toUTC();
}
} // namespace

ReportCenterController::ReportCenterController(QObject* parent)
    : QObject(parent)
{
    m_watcherDebounce.setInterval(500);
    m_watcherDebounce.setSingleShot(true);
    connect(&m_watcherDebounce, &QTimer::timeout, this, &ReportCenterController::handleWatcherTriggered);

    connect(&m_watcher, &QFileSystemWatcher::directoryChanged, this, &ReportCenterController::scheduleWatcherRefresh);
    connect(&m_watcher, &QFileSystemWatcher::fileChanged, this, &ReportCenterController::scheduleWatcherRefresh);
}

void ReportCenterController::setLastErrorMessage(const QString& message)
{
    if (!m_lastNotification.isEmpty()) {
        m_lastNotification.clear();
        Q_EMIT lastNotificationChanged();
    }
    if (m_lastError == message)
        return;
    m_lastError = message;
    Q_EMIT lastErrorChanged();
}

void ReportCenterController::clearLastErrorMessage()
{
    if (m_lastError.isEmpty())
        return;
    m_lastError.clear();
    Q_EMIT lastErrorChanged();
}

void ReportCenterController::setLastNotificationMessage(const QString& message)
{
    if (!m_lastError.isEmpty()) {
        m_lastError.clear();
        Q_EMIT lastErrorChanged();
    }
    if (m_lastNotification == message)
        return;
    m_lastNotification = message;
    Q_EMIT lastNotificationChanged();
}

void ReportCenterController::clearLastNotificationMessage()
{
    if (m_lastNotification.isEmpty())
        return;
    m_lastNotification.clear();
    Q_EMIT lastNotificationChanged();
}

void ReportCenterController::setPythonExecutable(const QString& executable)
{
    if (executable.trimmed().isEmpty())
        return;
    m_pythonExecutable = executable;
}

void ReportCenterController::setReportsDirectory(const QString& path)
{
    m_reportsDirectory = expandPath(path);
}

void ReportCenterController::setArchiveFormat(const QString& format)
{
    const QString normalized = normalizeArchiveFormat(format);
    if (m_archiveFormat == normalized)
        return;

    m_archiveFormat = normalized;
    Q_EMIT archiveFormatChanged();
}

void ReportCenterController::setRecentDaysFilter(int days)
{
    const int normalized = days < 0 ? 0 : days;
    if (m_recentDaysFilter == normalized)
        return;

    m_recentDaysFilter = normalized;
    Q_EMIT recentDaysFilterChanged();

    if (normalized > 0) {
        bool sinceChanged = false;
        bool untilChanged = false;
        if (m_sinceFilter.isValid()) {
            m_sinceFilter = {};
            sinceChanged = true;
        }
        if (m_untilFilter.isValid()) {
            m_untilFilter = {};
            untilChanged = true;
        }
        if (sinceChanged)
            Q_EMIT sinceFilterChanged();
        if (untilChanged)
            Q_EMIT untilFilterChanged();
    }

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::setSinceFilter(const QDateTime& since)
{
    QDateTime normalized = since;
    if (normalized.isValid())
        normalized = normalized.toUTC();

    if (sameInstant(m_sinceFilter, normalized))
        return;

    m_sinceFilter = normalized;
    Q_EMIT sinceFilterChanged();

    if (normalized.isValid() && m_recentDaysFilter != 0) {
        m_recentDaysFilter = 0;
        Q_EMIT recentDaysFilterChanged();
    }

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::setUntilFilter(const QDateTime& until)
{
    QDateTime normalized = until;
    if (normalized.isValid())
        normalized = normalized.toUTC();

    if (sameInstant(m_untilFilter, normalized))
        return;

    m_untilFilter = normalized;
    Q_EMIT untilFilterChanged();

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::setCategoryFilter(const QString& category)
{
    const QString normalized = category.trimmed();
    if (m_categoryFilter == normalized)
        return;

    m_categoryFilter = normalized;
    Q_EMIT categoryFilterChanged();

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::setSummaryStatusFilter(const QString& summaryStatus)
{
    QString normalized = summaryStatus.trimmed().toLower();
    if (normalized.isEmpty())
        normalized = QStringLiteral("any");
    static const QSet<QString> allowed = { QStringLiteral("any"), QStringLiteral("valid"), QStringLiteral("missing"), QStringLiteral("invalid") };
    if (!allowed.contains(normalized))
        normalized = QStringLiteral("any");

    if (m_summaryStatusFilter == normalized)
        return;

    m_summaryStatusFilter = normalized;
    Q_EMIT summaryStatusFilterChanged();

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::setExportsFilter(const QString& hasExports)
{
    QString normalized = hasExports.trimmed().toLower();
    if (normalized.isEmpty())
        normalized = QStringLiteral("any");
    static const QSet<QString> allowed = { QStringLiteral("any"), QStringLiteral("yes"), QStringLiteral("no") };
    if (!allowed.contains(normalized))
        normalized = QStringLiteral("any");

    if (m_exportsFilter == normalized)
        return;

    m_exportsFilter = normalized;
    Q_EMIT exportsFilterChanged();

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::setSearchQuery(const QString& query)
{
    const QString normalized = query.trimmed();
    if (m_searchQuery == normalized)
        return;

    m_searchQuery = normalized;
    Q_EMIT searchQueryChanged();

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::setLimit(int limit)
{
    const int normalized = limit < 0 ? 0 : limit;
    if (m_limit == normalized)
        return;

    m_limit = normalized;
    Q_EMIT limitChanged();

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::setOffset(int offset)
{
    const int normalized = offset < 0 ? 0 : offset;
    if (m_offset == normalized)
        return;

    m_offset = normalized;
    Q_EMIT offsetChanged();

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::setSortKey(const QString& sortKey)
{
    QString normalized = sortKey.trimmed().toLower();
    if (normalized.isEmpty())
        normalized = QStringLiteral("updated_at");
    static const QSet<QString> allowed = { QStringLiteral("updated_at"), QStringLiteral("created_at"), QStringLiteral("name"), QStringLiteral("size") };
    if (!allowed.contains(normalized))
        normalized = QStringLiteral("updated_at");

    if (m_sortKey == normalized)
        return;

    m_sortKey = normalized;
    Q_EMIT sortKeyChanged();

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::setSortDirection(const QString& sortDirection)
{
    QString normalized = sortDirection.trimmed().toLower();
    if (normalized != QStringLiteral("asc") && normalized != QStringLiteral("desc"))
        normalized = QStringLiteral("desc");

    if (m_sortDirection == normalized)
        return;

    m_sortDirection = normalized;
    Q_EMIT sortDirectionChanged();

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::clearSinceFilter()
{
    if (!m_sinceFilter.isValid())
        return;

    m_sinceFilter = {};
    Q_EMIT sinceFilterChanged();

    if (!m_busy)
        scheduleWatcherRefresh();
}

void ReportCenterController::clearUntilFilter()
{
    if (!m_untilFilter.isValid())
        return;

    m_untilFilter = {};
    Q_EMIT untilFilterChanged();

    if (!m_busy)
        scheduleWatcherRefresh();
}

bool ReportCenterController::refresh()
{
    if (m_busy)
        return false;

    clearLastErrorMessage();

    QStringList args;
    args << QStringLiteral("-m")
         << QStringLiteral("bot_core.reporting.ui_bridge")
         << QStringLiteral("overview");
    const QString reportsDir = resolveReportsDirectory();
    if (!reportsDir.isEmpty())
        args << QStringLiteral("--base-dir") << reportsDir;
    appendFilterArguments(args);

    beginTask();

    runBridge(args, [this](const BridgeResult& bridgeResult) {
        if (!bridgeResult.stderrData.isEmpty())
            qCWarning(lcReportCenter) << "report bridge stderr:" << QString::fromUtf8(bridgeResult.stderrData);

        bool result = false;
        if (bridgeResult.success)
            result = loadOverview(bridgeResult.stdoutData);

        if (!result) {
            if (!bridgeResult.success) {
                const QString message = bridgeResult.errorMessage.trimmed();
                if (!message.isEmpty()) {
                    setLastErrorMessage(message);
                } else if (m_lastError.isEmpty()) {
                    const QString stderrMessage = QString::fromUtf8(bridgeResult.stderrData).trimmed();
                    if (!stderrMessage.isEmpty())
                        setLastErrorMessage(stderrMessage);
                    else
                        setLastErrorMessage(tr("Nie udało się zbudować listy raportów"));
                }
            } else if (m_lastError.isEmpty()) {
                setLastErrorMessage(tr("Nie udało się zbudować listy raportów"));
            }
        }

        endTask();
        Q_EMIT overviewReady(result);
    });

    return true;
}

QVariantMap ReportCenterController::findReport(const QString& relativePath) const
{
    const QString normalized = relativePath.trimmed();
    for (const QVariant& entryVariant : m_reports) {
        const QVariantMap map = entryVariant.toMap();
        if (map.value(QStringLiteral("relative_path")).toString() == normalized)
            return map;
    }
    return {};
}

bool ReportCenterController::saveReportAs(const QString& relativePath, const QUrl& destinationUrl)
{
    const QVariantMap report = findReport(relativePath);
    if (report.isEmpty())
        return false;

    const QString absolutePath = report.value(QStringLiteral("absolute_path")).toString();
    if (absolutePath.isEmpty())
        return false;

    const QString destination = toFilePath(destinationUrl);
    if (destination.isEmpty()) {
        setLastErrorMessage(tr("Nieprawidłowa ścieżka docelowa eksportu raportu"));
        return false;
    }

    const QString normalizedSource = cleanPath(absolutePath);
    const QString normalizedDestination = cleanPath(destination);

    if (!normalizedSource.isEmpty() && normalizedSource == normalizedDestination) {
        setLastErrorMessage(tr("Ścieżka docelowa nie może być taka sama jak źródło"));
        return false;
    }

    const QString sourcePath = normalizedSource.isEmpty() ? absolutePath : normalizedSource;

    QFileInfo targetInfo(destination);
    QDir dir = targetInfo.dir();
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) {
        setLastErrorMessage(tr("Nie udało się utworzyć katalogu docelowego: %1").arg(dir.path()));
        return false;
    }

    if (QFile::exists(destination) && !QFile::remove(destination)) {
        setLastErrorMessage(tr("Nie można zastąpić istniejącego pliku: %1").arg(destination));
        return false;
    }

    if (!QFile::exists(sourcePath)) {
        setLastErrorMessage(tr("Plik źródłowy raportu nie istnieje: %1").arg(sourcePath));
        return false;
    }

    if (!QFile::copy(sourcePath, destination)) {
        setLastErrorMessage(tr("Kopiowanie raportu nie powiodło się (źródło: %1)").arg(sourcePath));
        return false;
    }

    clearLastErrorMessage();
    return true;
}

bool ReportCenterController::revealReport(const QString& relativePath)
{
    const QString normalized = relativePath.trimmed();
    if (normalized.isEmpty())
        return false;

    QVariantMap report = findReport(normalized);
    QString targetPath;
    if (!report.isEmpty()) {
        targetPath = report.value(QStringLiteral("absolute_path")).toString();
        if (targetPath.trimmed().isEmpty())
            targetPath = report.value(QStringLiteral("summary_path")).toString();
    } else {
        targetPath = normalized;
    }

    if (targetPath.trimmed().isEmpty())
        return false;

    QFileInfo info(targetPath);
    if (info.isFile())
        targetPath = info.absolutePath();
    else if (info.exists())
        targetPath = info.absoluteFilePath();
    else
        targetPath = QFileInfo(QDir::cleanPath(targetPath)).absoluteFilePath();

    if (targetPath.trimmed().isEmpty())
        return false;

    const bool opened = QDesktopServices::openUrl(QUrl::fromLocalFile(targetPath));
    if (!opened) {
        setLastErrorMessage(tr("Nie udało się otworzyć lokalizacji raportu: %1").arg(targetPath));
        return false;
    }

    clearLastErrorMessage();

    return true;
}

bool ReportCenterController::openExport(const QString& relativePath)
{
    const QString normalized = relativePath.trimmed();
    if (normalized.isEmpty())
        return false;

    QString absolutePath;
    for (const QVariant& reportVariant : m_reports) {
        const QVariantMap reportMap = reportVariant.toMap();
        const QVariantList exports = reportMap.value(QStringLiteral("exports")).toList();
        for (const QVariant& exportVariant : exports) {
            const QVariantMap exportMap = exportVariant.toMap();
            if (exportMap.value(QStringLiteral("relative_path")).toString() == normalized) {
                absolutePath = exportMap.value(QStringLiteral("absolute_path")).toString();
                break;
            }
        }
        if (!absolutePath.isEmpty())
            break;
    }

    if (absolutePath.trimmed().isEmpty()) {
        setLastErrorMessage(tr("Nie znaleziono eksportu: %1").arg(normalized));
        return false;
    }

    const bool opened = QDesktopServices::openUrl(QUrl::fromLocalFile(absolutePath));
    if (!opened) {
        setLastErrorMessage(tr("Nie udało się otworzyć eksportu: %1").arg(absolutePath));
        return false;
    }

    clearLastErrorMessage();

    return true;
}

bool ReportCenterController::previewDeleteReport(const QString& relativePath)
{
    const QString normalized = relativePath.trimmed();
    if (normalized.isEmpty()) {
        QVariantMap result;
        result.insert(QStringLiteral("status"), QStringLiteral("error"));
        result.insert(QStringLiteral("error"), tr("Niepoprawna ścieżka raportu"));
        Q_EMIT deletePreviewReady(normalized, result);
        return false;
    }

    QStringList args;
    args << QStringLiteral("-m")
         << QStringLiteral("bot_core.reporting.ui_bridge")
         << QStringLiteral("delete");

    const QString reportsDir = resolveReportsDirectory();
    if (!reportsDir.isEmpty())
        args << QStringLiteral("--base-dir") << reportsDir;

    args << QStringLiteral("--dry-run");
    args << normalized;

    beginTask();

    runBridge(args, [this, normalized](const BridgeResult& bridgeResult) {
        QVariantMap result;

        if (!bridgeResult.stderrData.isEmpty())
            qCWarning(lcReportCenter) << "report delete preview stderr:" << QString::fromUtf8(bridgeResult.stderrData).trimmed();

        if (!bridgeResult.success) {
            result.insert(QStringLiteral("status"), QStringLiteral("error"));
            QString message = bridgeResult.errorMessage.trimmed();
            if (message.isEmpty())
                message = QString::fromUtf8(bridgeResult.stderrData).trimmed();
            if (message.isEmpty())
                message = tr("Nie udało się przygotować podglądu usuwania");
            result.insert(QStringLiteral("error"), message);
        } else {
            QJsonParseError parseError{};
            const QJsonDocument doc = QJsonDocument::fromJson(bridgeResult.stdoutData, &parseError);
            if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
                result.insert(QStringLiteral("status"), QStringLiteral("error"));
                result.insert(QStringLiteral("error"), tr("Niepoprawna odpowiedź modułu raportów: %1").arg(parseError.errorString()));
            } else {
                result = doc.object().toVariantMap();
            }
        }

        Q_EMIT deletePreviewReady(normalized, result);
        endTask();
    });

    return true;
}

bool ReportCenterController::previewPurgeReports()
{
    QStringList args;
    args << QStringLiteral("-m")
         << QStringLiteral("bot_core.reporting.ui_bridge")
         << QStringLiteral("purge");

    const QString reportsDir = resolveReportsDirectory();
    if (!reportsDir.isEmpty())
        args << QStringLiteral("--base-dir") << reportsDir;

    appendFilterArguments(args);
    args << QStringLiteral("--dry-run");

    beginTask();

    runBridge(args, [this](const BridgeResult& bridgeResult) {
        QVariantMap result;

        if (!bridgeResult.stderrData.isEmpty())
            qCWarning(lcReportCenter) << "report purge preview stderr:" << QString::fromUtf8(bridgeResult.stderrData).trimmed();

        if (!bridgeResult.success) {
            result.insert(QStringLiteral("status"), QStringLiteral("error"));
            QString message = bridgeResult.errorMessage.trimmed();
            if (message.isEmpty())
                message = QString::fromUtf8(bridgeResult.stderrData).trimmed();
            if (message.isEmpty())
                message = tr("Nie udało się przygotować podglądu usuwania");
            result.insert(QStringLiteral("error"), message);
        } else {
            QJsonParseError parseError{};
            const QJsonDocument doc = QJsonDocument::fromJson(bridgeResult.stdoutData, &parseError);
            if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
                result.insert(QStringLiteral("status"), QStringLiteral("error"));
                result.insert(QStringLiteral("error"), tr("Niepoprawna odpowiedź modułu raportów: %1").arg(parseError.errorString()));
            } else {
                result = doc.object().toVariantMap();
            }
        }

        Q_EMIT purgePreviewReady(result);
        endTask();
    });

    return true;
}

bool ReportCenterController::deleteReport(const QString& relativePath)
{
    const QString normalized = relativePath.trimmed();
    if (normalized.isEmpty())
        return false;

    QStringList args;
    args << QStringLiteral("-m")
         << QStringLiteral("bot_core.reporting.ui_bridge")
         << QStringLiteral("delete");

    const QString reportsDir = resolveReportsDirectory();
    if (!reportsDir.isEmpty())
        args << QStringLiteral("--base-dir") << reportsDir;

    args << normalized;

    beginTask();

    runBridge(args, [this, normalized](const BridgeResult& bridgeResult) {
        if (!bridgeResult.stderrData.isEmpty())
            qCWarning(lcReportCenter) << "report delete stderr:" << QString::fromUtf8(bridgeResult.stderrData).trimmed();

        bool success = false;
        bool shouldRefresh = false;

        if (!bridgeResult.success) {
            QString message = bridgeResult.errorMessage.trimmed();
            if (message.isEmpty())
                message = QString::fromUtf8(bridgeResult.stderrData).trimmed();
            if (message.isEmpty())
                message = tr("Nie udało się usunąć raportu");
            setLastErrorMessage(message);
        } else {
            QJsonParseError parseError{};
            const QJsonDocument doc = QJsonDocument::fromJson(bridgeResult.stdoutData, &parseError);
            if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
                setLastErrorMessage(tr("Niepoprawna odpowiedź modułu raportów: %1").arg(parseError.errorString()));
            } else {
                const QJsonObject root = doc.object();
                const QString status = root.value(QStringLiteral("status")).toString();
                if (status != QStringLiteral("deleted")) {
                    QString message = root.value(QStringLiteral("error")).toString();
                    if (message.trimmed().isEmpty()) {
                        if (status == QStringLiteral("not_found"))
                            message = tr("Nie znaleziono raportu: %1").arg(normalized);
                        else if (status.trimmed().isEmpty())
                            message = tr("Niepoprawna odpowiedź modułu raportów");
                        else
                            message = tr("Nie udało się usunąć raportu");
                    }
                    setLastErrorMessage(message);
                } else {
                    const qint64 removedSize = root.value(QStringLiteral("removed_size")).toVariant().toLongLong();
                    const int removedFiles = root.value(QStringLiteral("removed_files")).toInt();
                    const int removedDirectories = root.value(QStringLiteral("removed_directories")).toInt();

                    QLocale locale;
                    QString formattedSize = locale.formattedDataSize(removedSize, 2, QLocale::DataSizeIecFormat);
                    if (formattedSize.trimmed().isEmpty())
                        formattedSize = locale.toString(removedSize);

                    clearLastErrorMessage();
                    setLastNotificationMessage(tr("Usunięto raport „%1” (pliki: %2, katalogi: %3, zwolniono %4).").arg(normalized)
                            .arg(removedFiles)
                            .arg(removedDirectories)
                            .arg(formattedSize));
                    success = true;
                    shouldRefresh = true;
                }
            }
        }

        endTask();

        if (shouldRefresh)
            refresh();

        Q_EMIT deleteFinished(normalized, success);
    });

    return true;
}

bool ReportCenterController::purgeReports()
{
    QStringList args;
    args << QStringLiteral("-m")
         << QStringLiteral("bot_core.reporting.ui_bridge")
         << QStringLiteral("purge");

    const QString reportsDir = resolveReportsDirectory();
    if (!reportsDir.isEmpty())
        args << QStringLiteral("--base-dir") << reportsDir;

    appendFilterArguments(args);

    beginTask();

    runBridge(args, [this](const BridgeResult& bridgeResult) {
        if (!bridgeResult.stderrData.isEmpty())
            qCWarning(lcReportCenter) << "report purge stderr:" << QString::fromUtf8(bridgeResult.stderrData).trimmed();

        bool success = false;
        bool shouldRefresh = false;

        if (!bridgeResult.success) {
            QString message = bridgeResult.errorMessage.trimmed();
            if (message.isEmpty())
                message = QString::fromUtf8(bridgeResult.stderrData).trimmed();
            if (message.isEmpty())
                message = tr("Nie udało się usunąć raportów");
            setLastErrorMessage(message);
        } else {
            QJsonParseError parseError{};
            const QJsonDocument doc = QJsonDocument::fromJson(bridgeResult.stdoutData, &parseError);
            if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
                setLastErrorMessage(tr("Niepoprawna odpowiedź modułu raportów: %1").arg(parseError.errorString()));
            } else {
                const QJsonObject root = doc.object();
                const QString status = root.value(QStringLiteral("status")).toString();

                const qint64 removedSize = root.value(QStringLiteral("removed_size")).toVariant().toLongLong();
                const int removedFiles = root.value(QStringLiteral("removed_files")).toInt();
                const int removedDirectories = root.value(QStringLiteral("removed_directories")).toInt();
                const int deletedCount = root.value(QStringLiteral("deleted_count")).toInt();
                const int plannedCount = root.value(QStringLiteral("planned_count")).toInt();

                if (status == QStringLiteral("empty")) {
                    clearLastErrorMessage();
                    setLastNotificationMessage(tr("Brak raportów do usunięcia dla aktywnych filtrów."));
                    success = true;
                } else {
                    QLocale locale;
                    QString formattedSize = locale.formattedDataSize(removedSize, 2, QLocale::DataSizeIecFormat);
                    if (formattedSize.trimmed().isEmpty())
                        formattedSize = locale.toString(removedSize);

                    if (status == QStringLiteral("completed")) {
                        clearLastErrorMessage();
                        setLastNotificationMessage(tr("Usunięto %1 raportów (pliki: %2, katalogi: %3, zwolniono %4).").arg(deletedCount)
                                .arg(removedFiles)
                                .arg(removedDirectories)
                                .arg(formattedSize));
                        success = true;
                        shouldRefresh = true;
                    } else if (status == QStringLiteral("partial_failure")) {
                        const QJsonArray errorsArray = root.value(QStringLiteral("errors")).toArray();
                        QStringList errors;
                        for (const QJsonValue& value : errorsArray) {
                            if (value.isString())
                                errors.append(value.toString());
                        }
                        QString message;
                        if (!errors.isEmpty())
                            message = errors.join(QLatin1String("; "));
                        if (message.trimmed().isEmpty())
                            message = tr("Niektóre raporty nie zostały usunięte.");
                        message = tr("Usunięto %1 z %2 raportów, ale część operacji zakończyła się błędem: %3")
                                      .arg(deletedCount)
                                      .arg(plannedCount)
                                      .arg(message);
                        setLastErrorMessage(message);
                        if (deletedCount > 0)
                            shouldRefresh = true;
                    } else if (status == QStringLiteral("error")) {
                        const QJsonArray errorsArray = root.value(QStringLiteral("errors")).toArray();
                        QStringList errors;
                        for (const QJsonValue& value : errorsArray) {
                            if (value.isString())
                                errors.append(value.toString());
                        }
                        QString message = errors.join(QLatin1String("; "));
                        if (message.trimmed().isEmpty())
                            message = tr("Nie udało się usunąć raportów");
                        setLastErrorMessage(message);
                    } else {
                        setLastErrorMessage(tr("Niepoprawna odpowiedź modułu raportów"));
                    }
                }
            }
        }

        endTask();

        if (shouldRefresh)
            refresh();

        Q_EMIT purgeFinished(success);
    });

    return true;
}


bool ReportCenterController::previewArchiveReports(const QString& destination, bool overwrite, const QString& format)
{
    QStringList args;
    args << QStringLiteral("-m")
         << QStringLiteral("bot_core.reporting.ui_bridge")
         << QStringLiteral("archive");

    const QString reportsDir = resolveReportsDirectory();
    if (!reportsDir.isEmpty())
        args << QStringLiteral("--base-dir") << reportsDir;

    const QString normalizedDestination = destination.trimmed();
    if (!normalizedDestination.isEmpty())
        args << QStringLiteral("--destination") << normalizedDestination;
    if (overwrite)
        args << QStringLiteral("--overwrite");

    const QString requestedFormat = format.trimmed().isEmpty() ? m_archiveFormat : format;
    const QString effectiveFormat = normalizeArchiveFormat(requestedFormat);
    setArchiveFormat(effectiveFormat);
    args << QStringLiteral("--format") << effectiveFormat;

    appendFilterArguments(args);
    args << QStringLiteral("--dry-run");

    beginTask();

    runBridge(args, [this, normalizedDestination, overwrite, effectiveFormat](const BridgeResult& bridgeResult) {
        QVariantMap result;

        if (!bridgeResult.stderrData.isEmpty())
            qCWarning(lcReportCenter) << "report archive preview stderr:" << QString::fromUtf8(bridgeResult.stderrData).trimmed();

        if (!bridgeResult.success) {
            result.insert(QStringLiteral("status"), QStringLiteral("error"));
            QString message = bridgeResult.errorMessage.trimmed();
            if (message.isEmpty())
                message = QString::fromUtf8(bridgeResult.stderrData).trimmed();
            if (message.isEmpty())
                message = tr("Nie udało się przygotować podglądu archiwizacji");
            result.insert(QStringLiteral("error"), message);
        } else {
            QJsonParseError parseError{};
            const QJsonDocument doc = QJsonDocument::fromJson(bridgeResult.stdoutData, &parseError);
            if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
                result.insert(QStringLiteral("status"), QStringLiteral("error"));
                result.insert(QStringLiteral("error"), tr("Niepoprawna odpowiedź modułu raportów: %1").arg(parseError.errorString()));
            } else {
                result = doc.object().toVariantMap();
            }
        }

        Q_EMIT archivePreviewReady(normalizedDestination, overwrite, effectiveFormat, result);
        endTask();
    });

    return true;
}


bool ReportCenterController::archiveReports(const QString& destination, bool overwrite, const QString& format)
{
    QStringList args;
    args << QStringLiteral("-m")
         << QStringLiteral("bot_core.reporting.ui_bridge")
         << QStringLiteral("archive");

    const QString reportsDir = resolveReportsDirectory();
    if (!reportsDir.isEmpty())
        args << QStringLiteral("--base-dir") << reportsDir;

    const QString normalizedDestination = destination.trimmed();
    if (!normalizedDestination.isEmpty())
        args << QStringLiteral("--destination") << normalizedDestination;
    if (overwrite)
        args << QStringLiteral("--overwrite");

    const QString requestedFormat = format.trimmed().isEmpty() ? m_archiveFormat : format;
    const QString effectiveFormat = normalizeArchiveFormat(requestedFormat);
    setArchiveFormat(effectiveFormat);
    args << QStringLiteral("--format") << effectiveFormat;

    appendFilterArguments(args);

    beginTask();

    runBridge(args, [this, normalizedDestination](const BridgeResult& bridgeResult) {
        if (!bridgeResult.stderrData.isEmpty())
            qCWarning(lcReportCenter) << "report archive stderr:" << QString::fromUtf8(bridgeResult.stderrData).trimmed();

        bool success = false;

        if (!bridgeResult.success) {
            QString message = bridgeResult.errorMessage.trimmed();
            if (message.isEmpty())
                message = QString::fromUtf8(bridgeResult.stderrData).trimmed();
            if (message.isEmpty())
                message = tr("Nie udało się zarchiwizować raportów");
            setLastErrorMessage(message);
        } else {
            QJsonParseError parseError{};
            const QJsonDocument doc = QJsonDocument::fromJson(bridgeResult.stdoutData, &parseError);
            if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
                setLastErrorMessage(tr("Niepoprawna odpowiedź modułu raportów: %1").arg(parseError.errorString()));
            } else {
                const QJsonObject root = doc.object();
                const QString status = root.value(QStringLiteral("status")).toString();
                const QString destinationDirectory = root.value(QStringLiteral("destination_directory")).toString();
                const qint64 copiedSize = root.value(QStringLiteral("copied_size")).toVariant().toLongLong();
                const int copiedFiles = root.value(QStringLiteral("copied_files")).toInt();
                const int copiedDirectories = root.value(QStringLiteral("copied_directories")).toInt();
                const int copiedCount = root.value(QStringLiteral("copied_count")).toInt();
                const int plannedCount = root.value(QStringLiteral("planned_count")).toInt();
                const QString responseFormat = root.value(QStringLiteral("format")).toString();
                const QString normalizedResponseFormat = normalizeArchiveFormat(responseFormat.isEmpty() ? m_archiveFormat : responseFormat);
                setArchiveFormat(normalizedResponseFormat);

                QString destinationLabel = destinationDirectory;
                if (destinationLabel.trimmed().isEmpty()) {
                    if (!normalizedDestination.isEmpty())
                        destinationLabel = normalizedDestination;
                    else
                        destinationLabel = defaultArchiveDestination();
                }

                QString formatLabel;
                if (normalizedResponseFormat == QLatin1String("zip"))
                    formatLabel = tr("ZIP");
                else if (normalizedResponseFormat == QLatin1String("tar"))
                    formatLabel = tr("TAR.GZ");
                else
                    formatLabel = tr("katalog docelowy");

                if (status == QStringLiteral("empty")) {
                    clearLastErrorMessage();
                    setLastNotificationMessage(tr("Brak raportów do archiwizacji dla aktywnych filtrów."));
                    success = true;
                } else {
                    QLocale locale;
                    QString formattedSize = locale.formattedDataSize(copiedSize, 2, QLocale::DataSizeIecFormat);
                    if (formattedSize.trimmed().isEmpty())
                        formattedSize = locale.toString(copiedSize);

                    if (status == QStringLiteral("completed")) {
                        clearLastErrorMessage();
                        setLastNotificationMessage(tr("Zarchiwizowano %1 raportów do „%2” (%3; pliki: %4, katalogi: %5, skopiowano %6).").arg(copiedCount)
                                .arg(destinationLabel)
                                .arg(formatLabel)
                                .arg(copiedFiles)
                                .arg(copiedDirectories)
                                .arg(formattedSize));
                        success = true;
                    } else if (status == QStringLiteral("partial_failure")) {
                        const QJsonArray errorsArray = root.value(QStringLiteral("errors")).toArray();
                        QStringList errors;
                        for (const QJsonValue& value : errorsArray) {
                            if (value.isString())
                                errors.append(value.toString());
                        }
                        QString message = errors.join(QLatin1String("; "));
                        if (message.trimmed().isEmpty())
                            message = tr("Niektóre raporty nie zostały zarchiwizowane.");
                        message = tr("Zarchiwizowano %1 z %2 raportów, ale część operacji zakończyła się błędem: %3")
                                      .arg(copiedCount)
                                      .arg(plannedCount)
                                      .arg(message);
                        setLastErrorMessage(message);
                    } else if (status == QStringLiteral("error")) {
                        const QJsonArray errorsArray = root.value(QStringLiteral("errors")).toArray();
                        QStringList errors;
                        for (const QJsonValue& value : errorsArray) {
                            if (value.isString())
                                errors.append(value.toString());
                        }
                        QString message = errors.join(QLatin1String("; "));
                        if (message.trimmed().isEmpty())
                            message = tr("Nie udało się zarchiwizować raportów");
                        setLastErrorMessage(message);
                    } else {
                        setLastErrorMessage(tr("Niepoprawna odpowiedź modułu raportów"));
                    }
                }
            }
        }

        endTask();
        Q_EMIT archiveFinished(success);
    });

    return true;
}


QString ReportCenterController::defaultArchiveDestination() const
{
    const QString baseDir = resolveReportsDirectory();
    QFileInfo baseInfo(baseDir);
    QString baseName = baseInfo.fileName();
    if (baseName.trimmed().isEmpty())
        baseName = QStringLiteral("reports");

    QDir baseDirectory(baseDir);
    QDir parentDirectory = baseDirectory;
    if (parentDirectory.cdUp())
        return cleanPath(parentDirectory.filePath(baseName + QStringLiteral("_archives")));

    return cleanPath(baseDirectory.filePath(baseName + QStringLiteral("_archives")));
}

void ReportCenterController::runBridge(const QStringList& arguments, BridgeCallback&& callback)
{
    auto sharedCallback = std::make_shared<BridgeCallback>(std::move(callback));
    auto* watcher = new QFutureWatcher<BridgeResult>(this);
    connect(watcher, &QFutureWatcher<BridgeResult>::finished, this, [watcher, sharedCallback]() {
        const BridgeResult result = watcher->result();
        watcher->deleteLater();
        (*sharedCallback)(result);
    });

    watcher->setFuture(QtConcurrent::run(&m_workerPool, [this, arguments]() {
        return executeBridge(arguments);
    }));
}

ReportCenterController::BridgeResult ReportCenterController::executeBridge(const QStringList& arguments) const
{
    BridgeResult result;

    if (m_pythonExecutable.trimmed().isEmpty()) {
        result.errorMessage = tr("Nie ustawiono interpretera Pythona dla modułu raportów");
        return result;
    }

    QProcess process;
    process.setProgram(m_pythonExecutable);
    process.setArguments(arguments);
    process.start();

    if (!process.waitForStarted()) {
        qCWarning(lcReportCenter) << "Nie udało się uruchomić modułu raportującego" << m_pythonExecutable
                                   << process.errorString();
        result.errorMessage = tr("Nie udało się uruchomić modułu raportów: %1").arg(process.errorString());
        return result;
    }

    if (!process.waitForFinished()) {
        qCWarning(lcReportCenter) << "Nie udało się uruchomić modułu raportującego" << m_pythonExecutable
                                   << process.errorString();
        result.stdoutData = process.readAllStandardOutput();
        result.stderrData = process.readAllStandardError();
        result.errorMessage = tr("Nie udało się uruchomić modułu raportów: %1").arg(process.errorString());
        return result;
    }

    result.stdoutData = process.readAllStandardOutput();
    result.stderrData = process.readAllStandardError();

    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        qCWarning(lcReportCenter) << "Bridge zakończył się kodem" << process.exitCode();
        result.errorMessage = tr("Moduł raportów zakończył się kodem %1").arg(process.exitCode());
        return result;
    }

    result.success = true;
    return result;
}

void ReportCenterController::beginTask()
{
    const bool wasIdle = (m_pendingTasks == 0);
    ++m_pendingTasks;
    if (wasIdle) {
        m_busy = true;
        Q_EMIT busyChanged();
    }
}

void ReportCenterController::endTask()
{
    if (m_pendingTasks <= 0) {
        m_pendingTasks = 0;
        if (m_busy) {
            m_busy = false;
            Q_EMIT busyChanged();
        }
        return;
    }

    --m_pendingTasks;
    if (m_pendingTasks == 0) {
        m_busy = false;
        Q_EMIT busyChanged();
    }
}

bool ReportCenterController::loadOverview(const QByteArray& data)
{
    QJsonParseError parseError{};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        setLastErrorMessage(tr("Niepoprawna odpowiedź modułu raportów: %1").arg(parseError.errorString()));
        return false;
    }

    const QJsonObject root = doc.object();
    const QString baseDirectory = root.value(QStringLiteral("base_directory")).toString();
    const QJsonArray reportsArray = root.value(QStringLiteral("reports")).toArray();
    QVariantList reports;
    reports.reserve(reportsArray.size());
    for (const QJsonValue& value : reportsArray)
        reports.append(value.toVariant());

    m_reports = reports;
    Q_EMIT reportsChanged();

    const QJsonArray categoriesArray = root.value(QStringLiteral("categories")).toArray();
    QVariantList categories;
    categories.reserve(categoriesArray.size());
    for (const QJsonValue& value : categoriesArray) {
        if (value.isObject()) {
            categories.append(value.toObject().toVariantMap());
            continue;
        }
        if (value.isString()) {
            const QString identifier = value.toString();
            QVariantMap fallback;
            fallback.insert(QStringLiteral("id"), identifier);
            fallback.insert(QStringLiteral("label"), identifier);
            categories.append(fallback);
        }
    }
    if (m_categories != categories) {
        m_categories = categories;
        Q_EMIT categoriesChanged();
    }

    QVariantMap overview;
    const QJsonValue summaryValue = root.value(QStringLiteral("summary"));
    if (summaryValue.isObject())
        overview = summaryValue.toObject().toVariantMap();
    else if (!summaryValue.isUndefined() && !summaryValue.isNull())
        overview.insert(QStringLiteral("value"), summaryValue.toVariant());

    QVariantMap pagination;
    const QJsonValue paginationValue = root.value(QStringLiteral("pagination"));
    if (paginationValue.isObject())
        pagination = paginationValue.toObject().toVariantMap();
    else if (!paginationValue.isUndefined() && !paginationValue.isNull())
        pagination.insert(QStringLiteral("value"), paginationValue.toVariant());


    QString filterCategory;
    QString filterSummaryStatus = QStringLiteral("any");
    QString filterExports = QStringLiteral("any");
    int filterLimit = 0;
    int filterOffset = 0;
    QString filterSortKey = QStringLiteral("updated_at");
    QString filterSortDirection = QStringLiteral("desc");
    QString filterQuery;
    QDateTime filterSince;
    QDateTime filterUntil;
    const QJsonValue filtersValue = root.value(QStringLiteral("filters"));
    if (filtersValue.isObject()) {
        const QJsonObject filtersObject = filtersValue.toObject();
        const QJsonValue categoriesValue = filtersObject.value(QStringLiteral("categories"));
        if (categoriesValue.isArray()) {
            const QJsonArray categoriesArray = categoriesValue.toArray();
            if (!categoriesArray.isEmpty())
                filterCategory = categoriesArray.at(0).toString();
        } else if (categoriesValue.isString()) {
            filterCategory = categoriesValue.toString();
        }

        const QJsonValue sinceValue = filtersObject.value(QStringLiteral("since"));
        if (sinceValue.isString()) {
            const QDateTime parsed = QDateTime::fromString(sinceValue.toString(), Qt::ISODate);
            if (parsed.isValid())
                filterSince = parsed.toUTC();
        }

        const QJsonValue untilValue = filtersObject.value(QStringLiteral("until"));
        if (untilValue.isString()) {
            const QDateTime parsed = QDateTime::fromString(untilValue.toString(), Qt::ISODate);
            if (parsed.isValid())
                filterUntil = parsed.toUTC();
        }

        const QJsonValue summaryStatusValue = filtersObject.value(QStringLiteral("summary_status"));
        if (summaryStatusValue.isString())
            filterSummaryStatus = summaryStatusValue.toString();

        const QJsonValue exportsValue = filtersObject.value(QStringLiteral("has_exports"));
        if (exportsValue.isString())
            filterExports = exportsValue.toString();

        const QJsonValue limitValue = filtersObject.value(QStringLiteral("limit"));
        if (limitValue.isDouble())
            filterLimit = qMax(0, limitValue.toInt());
        const QJsonValue offsetValue = filtersObject.value(QStringLiteral("offset"));
        if (offsetValue.isDouble())
            filterOffset = qMax(0, offsetValue.toInt());
        const QJsonValue sortKeyValue = filtersObject.value(QStringLiteral("sort_key"));
        if (sortKeyValue.isString())
            filterSortKey = sortKeyValue.toString();
        const QJsonValue sortDirectionValue = filtersObject.value(QStringLiteral("sort_direction"));
        if (sortDirectionValue.isString())
            filterSortDirection = sortDirectionValue.toString();
        const QJsonValue queryValue = filtersObject.value(QStringLiteral("query"));
        if (queryValue.isString())
            filterQuery = queryValue.toString();
    }

    if (m_recentDaysFilter <= 0) {
        if (!sameInstant(m_sinceFilter, filterSince)) {
            m_sinceFilter = filterSince;
            Q_EMIT sinceFilterChanged();
        }
    } else if (m_sinceFilter.isValid()) {
        m_sinceFilter = {};
        Q_EMIT sinceFilterChanged();
    }

    if (!sameInstant(m_untilFilter, filterUntil)) {
        m_untilFilter = filterUntil;
        Q_EMIT untilFilterChanged();
    }

    if (m_categoryFilter != filterCategory) {
        m_categoryFilter = filterCategory;
        Q_EMIT categoryFilterChanged();
    }
    QString normalizedSummary = filterSummaryStatus.trimmed().toLower();
    if (normalizedSummary.isEmpty())
        normalizedSummary = QStringLiteral("any");
    static const QSet<QString> allowed = { QStringLiteral("any"), QStringLiteral("valid"), QStringLiteral("missing"), QStringLiteral("invalid") };
    if (!allowed.contains(normalizedSummary))
        normalizedSummary = QStringLiteral("any");
    if (m_summaryStatusFilter != normalizedSummary) {
        m_summaryStatusFilter = normalizedSummary;
        Q_EMIT summaryStatusFilterChanged();
    }

    QString normalizedExports = filterExports.trimmed().toLower();
    if (normalizedExports.isEmpty())
        normalizedExports = QStringLiteral("any");
    static const QSet<QString> allowedExports = { QStringLiteral("any"), QStringLiteral("yes"), QStringLiteral("no") };
    if (!allowedExports.contains(normalizedExports))
        normalizedExports = QStringLiteral("any");
    if (m_exportsFilter != normalizedExports) {
        m_exportsFilter = normalizedExports;
        Q_EMIT exportsFilterChanged();
    }
    if (m_limit != filterLimit) {
        m_limit = filterLimit;
        Q_EMIT limitChanged();
    }
    if (m_offset != filterOffset) {
        m_offset = filterOffset;
        Q_EMIT offsetChanged();
    }

    QString normalizedSortKey = filterSortKey.trimmed().toLower();
    static const QSet<QString> allowedSortKeys = { QStringLiteral("updated_at"), QStringLiteral("created_at"), QStringLiteral("name"), QStringLiteral("size") };
    if (!allowedSortKeys.contains(normalizedSortKey))
        normalizedSortKey = QStringLiteral("updated_at");
    if (m_sortKey != normalizedSortKey) {
        m_sortKey = normalizedSortKey;
        Q_EMIT sortKeyChanged();
    }

    QString normalizedDirection = filterSortDirection.trimmed().toLower();
    if (normalizedDirection != QStringLiteral("asc") && normalizedDirection != QStringLiteral("desc"))
        normalizedDirection = QStringLiteral("desc");
    if (m_sortDirection != normalizedDirection) {
        m_sortDirection = normalizedDirection;
        Q_EMIT sortDirectionChanged();
    }
    const QString normalizedQuery = filterQuery.trimmed();
    if (m_searchQuery != normalizedQuery) {
        m_searchQuery = normalizedQuery;
        Q_EMIT searchQueryChanged();
    }
    if (m_overviewStats != overview) {
        m_overviewStats = overview;
        Q_EMIT overviewStatsChanged();
    }

    if (m_overviewPagination != pagination) {
        m_overviewPagination = pagination;
        Q_EMIT overviewPaginationChanged();
    }

    rebuildWatcher(baseDirectory, reports);

    clearLastErrorMessage();
    return true;
}

QString ReportCenterController::resolveReportsDirectory() const
{
    if (!m_reportsDirectory.trimmed().isEmpty())
        return m_reportsDirectory;
    return expandPath(QStringLiteral("var/reports"));
}

QString ReportCenterController::expandPath(const QString& path)
{
    return cleanPath(bot::shell::utils::expandPath(path));
}

void ReportCenterController::appendFilterArguments(QStringList& args, bool includePagination) const
{
    if (m_sinceFilter.isValid()) {
        args << QStringLiteral("--since") << m_sinceFilter.toUTC().toString(Qt::ISODate);
    } else if (m_recentDaysFilter > 0) {
        const QDateTime since = QDateTime::currentDateTimeUtc().addDays(-m_recentDaysFilter);
        args << QStringLiteral("--since") << since.toString(Qt::ISODate);
    }

    if (m_untilFilter.isValid())
        args << QStringLiteral("--until") << m_untilFilter.toUTC().toString(Qt::ISODate);

    if (!m_categoryFilter.trimmed().isEmpty())
        args << QStringLiteral("--category") << m_categoryFilter.trimmed();

    if (!m_summaryStatusFilter.isEmpty() && m_summaryStatusFilter != QStringLiteral("any"))
        args << QStringLiteral("--summary-status") << m_summaryStatusFilter;

    if (!m_exportsFilter.isEmpty() && m_exportsFilter != QStringLiteral("any"))
        args << QStringLiteral("--has-exports") << m_exportsFilter;

    if (includePagination) {
        if (m_offset > 0)
            args << QStringLiteral("--offset") << QString::number(m_offset);
        if (m_limit > 0)
            args << QStringLiteral("--limit") << QString::number(m_limit);
    }

    if (!m_sortKey.isEmpty() && m_sortKey != QStringLiteral("updated_at"))
        args << QStringLiteral("--sort") << m_sortKey;

    if (!m_sortDirection.isEmpty() && m_sortDirection != QStringLiteral("desc"))
        args << QStringLiteral("--sort-direction") << m_sortDirection;

    if (!m_searchQuery.trimmed().isEmpty())
        args << QStringLiteral("--query") << m_searchQuery.trimmed();
}

void ReportCenterController::rebuildWatcher(const QString& baseDirectory, const QVariantList& reports)
{
    QString normalizedBase = cleanPath(baseDirectory);
    if (normalizedBase.isEmpty()) {
        if (!m_watcher.directories().isEmpty())
            m_watcher.removePaths(m_watcher.directories());
        if (!m_watcher.files().isEmpty())
            m_watcher.removePaths(m_watcher.files());
        return;
    }

    QSet<QString> directoriesToWatch;
    QFileInfo baseInfo(normalizedBase);
    if (baseInfo.exists()) {
        if (baseInfo.isDir())
            directoriesToWatch.insert(baseInfo.absoluteFilePath());
        else
            directoriesToWatch.insert(baseInfo.dir().absolutePath());
    } else {
        const QString parentDir = baseInfo.dir().absolutePath();
        if (!parentDir.isEmpty())
            directoriesToWatch.insert(parentDir);
    }

    const auto addDirectoryCandidate = [&directoriesToWatch](const QString& candidate) {
        const QString trimmed = candidate.trimmed();
        if (trimmed.isEmpty())
            return;

        QFileInfo info(trimmed);
        QString directoryPath;
        if (info.exists() && info.isDir())
            directoryPath = info.absoluteFilePath();
        else
            directoryPath = info.dir().absolutePath();

        if (!directoryPath.isEmpty())
            directoriesToWatch.insert(QDir::cleanPath(directoryPath));
    };

    const QDir baseDir(normalizedBase);

    for (const QVariant& entryVariant : reports) {
        const QVariantMap entry = entryVariant.toMap();
        const QString relativePath = entry.value(QStringLiteral("relative_path")).toString();
        if (relativePath.trimmed().isEmpty())
            continue;
        const QString combined = cleanPath(baseDir.filePath(relativePath));
        if (!combined.isEmpty())
            addDirectoryCandidate(combined);

        const QString summaryPath = cleanPath(entry.value(QStringLiteral("summary_path")).toString());
        if (!summaryPath.isEmpty())
            addDirectoryCandidate(summaryPath);

        const QVariantList exports = entry.value(QStringLiteral("exports")).toList();
        for (const QVariant& exportVariant : exports) {
            const QVariantMap exportMap = exportVariant.toMap();
            const QString exportPath = cleanPath(exportMap.value(QStringLiteral("absolute_path")).toString());
            if (!exportPath.isEmpty())
                addDirectoryCandidate(exportPath);
        }
    }

    QStringList toWatch;
    toWatch.reserve(directoriesToWatch.size());
    for (auto it = directoriesToWatch.cbegin(); it != directoriesToWatch.cend(); ++it) {
        const QFileInfo info(*it);
        if (!info.exists() || !info.isDir())
            continue;
        toWatch.append(info.absoluteFilePath());
    }

    std::sort(toWatch.begin(), toWatch.end());

    if (toWatch.size() > kWatcherSoftLimit) {
        qCWarning(lcReportCenter)
            << "Report watcher is tracking" << toWatch.size()
            << "directories; consider archiving or relocating old reports to reduce load";
    }

    QSignalBlocker blocker(&m_watcher);
    QStringList directories = m_watcher.directories();
    QStringList files = m_watcher.files();
    if (!directories.isEmpty())
        m_watcher.removePaths(directories);
    if (!files.isEmpty())
        m_watcher.removePaths(files);
    if (!toWatch.isEmpty())
        m_watcher.addPaths(toWatch);
}

void ReportCenterController::scheduleWatcherRefresh()
{
    m_watcherDebounce.start();
}

void ReportCenterController::handleWatcherTriggered()
{
    if (m_busy) {
        m_watcherDebounce.start();
        return;
    }
    refresh();
}
