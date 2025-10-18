#include "ReportCenterController.hpp"

#include <QByteArray>
#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLoggingCategory>
#include <QProcess>
#include <QVariantMap>

Q_LOGGING_CATEGORY(lcReportCenter, "bot.shell.reporting.center")

ReportCenterController::ReportCenterController(QObject* parent)
    : QObject(parent)
{
}

void ReportCenterController::setPythonExecutable(const QString& executable)
{
    if (executable.trimmed().isEmpty()) {
        return;
    }
    m_pythonExecutable = executable;
}

void ReportCenterController::setReportsRoot(const QString& root)
{
    if (root.trimmed().isEmpty()) {
        m_reportsRoot.clear();
        return;
    }
    if (root.startsWith(QStringLiteral("~/"))) {
        m_reportsRoot = QDir::homePath() + root.mid(1);
        return;
    }
    if (root == QStringLiteral("~")) {
        m_reportsRoot = QDir::homePath();
        return;
    }
    m_reportsRoot = root;
}

bool ReportCenterController::refresh()
{
    if (m_busy) {
        return false;
    }

    m_busy = true;
    Q_EMIT busyChanged();

    QStringList args;
    args << QStringLiteral("-m") << QStringLiteral("bot_core.reporting.ui_bridge") << QStringLiteral("list");
    if (!m_reportsRoot.isEmpty()) {
        args << QStringLiteral("--root") << m_reportsRoot;
    }

    QByteArray stdoutData;
    QByteArray stderrData;
    const bool ok = runBridge(args, &stdoutData, &stderrData);
    if (!stderrData.isEmpty()) {
        qCWarning(lcReportCenter) << "Bridge stderr:" << QString::fromUtf8(stderrData);
    }

    bool parsed = false;
    if (!stdoutData.isEmpty()) {
        parsed = loadReportsFromJson(stdoutData);
    }

    if (!ok) {
        if (!parsed) {
            qCWarning(lcReportCenter) << "Bridge list command failed";
        }
        m_busy = false;
        Q_EMIT busyChanged();
        return false;
    }

    if (!parsed) {
        m_busy = false;
        Q_EMIT busyChanged();
        return false;
    }

    m_busy = false;
    Q_EMIT busyChanged();
    return true;
}

bool ReportCenterController::deleteReport(const QString& path)
{
    if (m_busy) {
        return false;
    }

    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty()) {
        qCWarning(lcReportCenter) << "Odmowa usunięcia raportu – pusta ścieżka";
        return false;
    }

    m_busy = true;
    Q_EMIT busyChanged();

    QStringList args;
    args << QStringLiteral("-m") << QStringLiteral("bot_core.reporting.ui_bridge") << QStringLiteral("delete") << trimmed;
    if (!m_reportsRoot.isEmpty()) {
        args << QStringLiteral("--root") << m_reportsRoot;
    }

    QByteArray stdoutData;
    QByteArray stderrData;
    const bool ok = runBridge(args, &stdoutData, &stderrData);
    if (!stderrData.isEmpty()) {
        qCWarning(lcReportCenter) << "Bridge stderr:" << QString::fromUtf8(stderrData);
    }

    QString status;
    QString resolvedPath = trimmed;
    QString errorMessage;

    if (!stdoutData.isEmpty()) {
        QJsonParseError parseError{};
        const QJsonDocument doc = QJsonDocument::fromJson(stdoutData, &parseError);
        if (parseError.error != QJsonParseError::NoError) {
            qCWarning(lcReportCenter) << "Niepoprawna odpowiedź bridge delete" << parseError.errorString();
        } else if (doc.isObject()) {
            const QJsonObject obj = doc.object();
            status = obj.value(QStringLiteral("status")).toString();
            resolvedPath = obj.value(QStringLiteral("path")).toString(trimmed);
            errorMessage = obj.value(QStringLiteral("reason")).toString();
        }
    }

    bool shouldRefresh = false;
    if (status.compare(QStringLiteral("ok"), Qt::CaseInsensitive) == 0) {
        shouldRefresh = true;
        Q_EMIT reportDeleted(resolvedPath);
    } else if (status.compare(QStringLiteral("not_found"), Qt::CaseInsensitive) == 0) {
        Q_EMIT reportOperationFailed(tr("Raport %1 nie istnieje").arg(resolvedPath));
    } else if (status.compare(QStringLiteral("forbidden"), Qt::CaseInsensitive) == 0) {
        const QString message = errorMessage.isEmpty() ? tr("Raport znajduje się poza katalogiem magazynu") : errorMessage;
        Q_EMIT reportOperationFailed(message);
    } else if (!status.isEmpty()) {
        const QString message = errorMessage.isEmpty() ? tr("Bridge zwrócił status %1").arg(status) : errorMessage;
        Q_EMIT reportOperationFailed(message);
    } else if (!ok) {
        Q_EMIT reportOperationFailed(tr("Usunięcie raportu nie powiodło się"));
    }

    m_busy = false;
    Q_EMIT busyChanged();

    if (shouldRefresh) {
        refresh();
    }

    return ok && status.compare(QStringLiteral("ok"), Qt::CaseInsensitive) == 0;
}

bool ReportCenterController::runBridge(const QStringList& arguments,
                                       QByteArray* stdoutData,
                                       QByteArray* stderrData) const
{
    QProcess process;
    process.setProgram(m_pythonExecutable);
    process.setArguments(arguments);
    process.start();
    if (!process.waitForFinished()) {
        qCWarning(lcReportCenter) << "Nie udało się uruchomić bridge" << m_pythonExecutable << process.errorString();
        return false;
    }
    if (stdoutData) {
        *stdoutData = process.readAllStandardOutput();
    }
    if (stderrData) {
        *stderrData = process.readAllStandardError();
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        qCWarning(lcReportCenter) << "Bridge zakończył się kodem" << process.exitCode();
        return false;
    }
    return true;
}

bool ReportCenterController::loadReportsFromJson(const QByteArray& data)
{
    QJsonParseError parseError{};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        qCWarning(lcReportCenter) << "Nie udało się sparsować JSON bridge:" << parseError.errorString();
        return false;
    }

    const QJsonObject root = doc.object();
    const QJsonArray reportsArray = root.value(QStringLiteral("reports")).toArray();
    QVariantList reports;
    reports.reserve(reportsArray.size());
    for (const QJsonValue& value : reportsArray) {
        if (!value.isObject()) {
            continue;
        }
        QVariantMap map;
        const QJsonObject obj = value.toObject();
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            map.insert(it.key(), it.value().toVariant());
        }
        reports.append(map);
    }
    m_reports = reports;
    Q_EMIT reportsChanged();
    return true;
}
