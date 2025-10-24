#include "SupportBundleController.hpp"

#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>

#include "utils/PathUtils.hpp"

namespace {

Q_LOGGING_CATEGORY(lcSupportBundle, "bot.shell.support")

QString normalizePath(const QString& raw)
{
    QString path = raw.trimmed();
    if (path.isEmpty())
        return path;
    if (path.startsWith(QStringLiteral("~/")))
        return QDir::homePath() + path.mid(1);
    if (path == QStringLiteral("~"))
        return QDir::homePath();
    return path;
}

} // namespace

SupportBundleController::SupportBundleController(QObject* parent)
    : QObject(parent)
{
    const QDir cwd = QDir::current();
    m_logsPath = cwd.absoluteFilePath(QStringLiteral("logs"));
    m_reportsPath = cwd.absoluteFilePath(QStringLiteral("var/reports"));
    m_licensesPath = cwd.absoluteFilePath(QStringLiteral("var/licenses"));
    m_metricsPath = cwd.absoluteFilePath(QStringLiteral("var/metrics"));
    m_auditPath = cwd.absoluteFilePath(QStringLiteral("var/audit"));
    m_outputDirectory = cwd.absoluteFilePath(QStringLiteral("var/support"));
    m_metadata.insert(QStringLiteral("origin"), QStringLiteral("desktop_ui"));
}

SupportBundleController::~SupportBundleController()
{
    resetActiveProcess();
}

void SupportBundleController::setPythonExecutable(const QString& executable)
{
    const QString expanded = expandPath(executable);
    if (expanded == m_pythonExecutable)
        return;
    m_pythonExecutable = expanded.isEmpty() ? QStringLiteral("python3") : expanded;
    Q_EMIT pythonExecutableChanged();
}

void SupportBundleController::setScriptPath(const QString& path)
{
    const QString expanded = expandPath(path);
    if (expanded == m_scriptPath)
        return;
    m_scriptPath = expanded;
    Q_EMIT scriptPathChanged();
}

void SupportBundleController::setOutputDirectory(const QString& directory)
{
    const QString expanded = expandPath(directory);
    if (expanded == m_outputDirectory)
        return;
    m_outputDirectory = expanded;
    Q_EMIT outputDirectoryChanged();
}

void SupportBundleController::setDefaultBasename(const QString& basename)
{
    QString trimmed = basename.trimmed();
    if (trimmed.isEmpty())
        trimmed = QStringLiteral("support-bundle");
    if (trimmed == m_defaultBasename)
        return;
    m_defaultBasename = trimmed;
    Q_EMIT defaultBasenameChanged();
}

void SupportBundleController::setFormat(const QString& format)
{
    const QString sanitized = sanitizeFormat(format);
    if (sanitized == m_format)
        return;
    m_format = sanitized;
    Q_EMIT formatChanged();
}

void SupportBundleController::setIncludeLogs(bool include)
{
    if (m_includeLogs == include)
        return;
    m_includeLogs = include;
    Q_EMIT includeLogsChanged();
}

void SupportBundleController::setIncludeReports(bool include)
{
    if (m_includeReports == include)
        return;
    m_includeReports = include;
    Q_EMIT includeReportsChanged();
}

void SupportBundleController::setIncludeLicenses(bool include)
{
    if (m_includeLicenses == include)
        return;
    m_includeLicenses = include;
    Q_EMIT includeLicensesChanged();
}

void SupportBundleController::setIncludeMetrics(bool include)
{
    if (m_includeMetrics == include)
        return;
    m_includeMetrics = include;
    Q_EMIT includeMetricsChanged();
}

void SupportBundleController::setIncludeAudit(bool include)
{
    if (m_includeAudit == include)
        return;
    m_includeAudit = include;
    Q_EMIT includeAuditChanged();
}

void SupportBundleController::setLogsPath(const QString& path)
{
    const QString expanded = expandPath(path);
    if (expanded == m_logsPath)
        return;
    m_logsPath = expanded;
    Q_EMIT logsPathChanged();
}

void SupportBundleController::setReportsPath(const QString& path)
{
    const QString expanded = expandPath(path);
    if (expanded == m_reportsPath)
        return;
    m_reportsPath = expanded;
    Q_EMIT reportsPathChanged();
}

void SupportBundleController::setLicensesPath(const QString& path)
{
    const QString expanded = expandPath(path);
    if (expanded == m_licensesPath)
        return;
    m_licensesPath = expanded;
    Q_EMIT licensesPathChanged();
}

void SupportBundleController::setMetricsPath(const QString& path)
{
    const QString expanded = expandPath(path);
    if (expanded == m_metricsPath)
        return;
    m_metricsPath = expanded;
    Q_EMIT metricsPathChanged();
}

void SupportBundleController::setAuditPath(const QString& path)
{
    const QString expanded = expandPath(path);
    if (expanded == m_auditPath)
        return;
    m_auditPath = expanded;
    Q_EMIT auditPathChanged();
}

void SupportBundleController::setMetadata(const QVariantMap& metadata)
{
    QVariantMap normalized = metadata;
    if (!normalized.contains(QStringLiteral("origin")))
        normalized.insert(QStringLiteral("origin"), QStringLiteral("desktop_ui"));

    if (normalized == m_metadata)
        return;

    m_metadata = normalized;
    Q_EMIT metadataChanged();
}

void SupportBundleController::setExtraIncludeSpecs(const QStringList& specs)
{
    m_extraIncludeSpecs = specs;
}

QString SupportBundleController::expandPath(const QString& path) const
{
    return bot::shell::utils::expandPath(normalizePath(path));
}

QString SupportBundleController::sanitizeFormat(const QString& requested) const
{
    const QString normalized = requested.trimmed().toLower();
    if (normalized == QStringLiteral("zip"))
        return QStringLiteral("zip");
    return QStringLiteral("tar.gz");
}

QString SupportBundleController::buildDefaultDestination() const
{
    return QDir(outputDirectory()).absolutePath();
}

QStringList SupportBundleController::buildIncludeArguments() const
{
    QStringList arguments;
    const auto appendInclude = [&arguments](const QString& label, const QString& path) {
        if (label.isEmpty() || path.trimmed().isEmpty())
            return;
        arguments << QStringLiteral("--include") << QStringLiteral("%1=%2").arg(label, path);
    };

    if (m_includeLogs)
        appendInclude(QStringLiteral("logs"), m_logsPath);
    if (m_includeReports)
        appendInclude(QStringLiteral("reports"), m_reportsPath);
    if (m_includeLicenses)
        appendInclude(QStringLiteral("licenses"), m_licensesPath);
    if (m_includeMetrics)
        appendInclude(QStringLiteral("metrics"), m_metricsPath);
    if (m_includeAudit)
        appendInclude(QStringLiteral("audit"), m_auditPath);

    return arguments;
}

QStringList SupportBundleController::buildDisableArguments() const
{
    QStringList arguments;
    const auto appendDisable = [&arguments](bool includeFlag, const QString& label) {
        if (includeFlag)
            return;
        arguments << QStringLiteral("--disable") << label;
    };

    appendDisable(m_includeLogs, QStringLiteral("logs"));
    appendDisable(m_includeReports, QStringLiteral("reports"));
    appendDisable(m_includeLicenses, QStringLiteral("licenses"));
    appendDisable(m_includeMetrics, QStringLiteral("metrics"));
    appendDisable(m_includeAudit, QStringLiteral("audit"));

    return arguments;
}

QStringList SupportBundleController::buildMetadataArguments() const
{
    QStringList arguments;
    for (auto it = m_metadata.constBegin(); it != m_metadata.constEnd(); ++it) {
        const QString key = it.key().trimmed();
        if (key.isEmpty())
            continue;
        arguments << QStringLiteral("--metadata")
                  << QStringLiteral("%1=%2").arg(key, it.value().toString());
    }
    return arguments;
}

QStringList SupportBundleController::buildExtraIncludeArguments() const
{
    QStringList arguments;
    for (const QString& spec : m_extraIncludeSpecs) {
        const QString trimmed = spec.trimmed();
        if (trimmed.isEmpty())
            continue;
        arguments << QStringLiteral("--include") << trimmed;
    }
    return arguments;
}

QStringList SupportBundleController::buildCommandArguments(const QUrl& destination, bool dryRun) const
{
    QStringList args;

    if (!destination.isEmpty()) {
        if (destination.isLocalFile())
            args << QStringLiteral("--output") << destination.toLocalFile();
        else
            args << QStringLiteral("--output") << destination.toString();
    } else if (!m_outputDirectory.trimmed().isEmpty()) {
        args << QStringLiteral("--output-dir") << m_outputDirectory;
    }

    args << QStringLiteral("--basename") << m_defaultBasename;
    args << QStringLiteral("--format") << m_format;

    args.append(buildIncludeArguments());
    args.append(buildDisableArguments());
    args.append(buildExtraIncludeArguments());
    args.append(buildMetadataArguments());

    if (dryRun)
        args << QStringLiteral("--dry-run");

    return args;
}

bool SupportBundleController::exportBundle(const QUrl& destination, bool dryRun)
{
    if (m_busy) {
        qCWarning(lcSupportBundle) << "Attempted to start support bundle export while busy";
        return false;
    }
    if (m_scriptPath.trimmed().isEmpty()) {
        qCWarning(lcSupportBundle) << "Support bundle script path is not configured";
        return false;
    }

    QString program = m_pythonExecutable.trimmed();
    if (program.isEmpty())
        program = QStringLiteral("python3");

    QStringList args = buildCommandArguments(destination, dryRun);
    args.prepend(m_scriptPath);

    if (m_process) {
        m_process->disconnect(this);
        m_process.reset();
    }
    m_process = std::make_unique<QProcess>(this);
    m_process->setProgram(program);
    m_process->setArguments(args);

    connect(m_process.get(), qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            &SupportBundleController::handleProcessFinished);

    m_lastErrorMessage.clear();
    m_lastStatusMessage.clear();

    m_busy = true;
    Q_EMIT busyChanged();

    qCDebug(lcSupportBundle) << "Starting support bundle export" << program << args;
    m_process->start();
    if (!m_process->waitForStarted()) {
        qCWarning(lcSupportBundle) << "Failed to start support bundle process" << program << m_process->errorString();
        m_lastErrorMessage = tr("Nie udało się uruchomić procesu eksportu pakietu wsparcia: %1")
                                   .arg(m_process->errorString());
        m_lastResult = {
            {QStringLiteral("status"), QStringLiteral("error")},
            {QStringLiteral("error"), m_lastErrorMessage},
        };
        Q_EMIT lastErrorMessageChanged();
        Q_EMIT lastResultChanged();
        resetActiveProcess();
        m_busy = false;
        Q_EMIT busyChanged();
        Q_EMIT exportFinished(false, m_lastResult);
        return false;
    }

    return true;
}

void SupportBundleController::handleProcessFinished(int exitCode, QProcess::ExitStatus status)
{
    if (!m_process)
        return;

    const QByteArray stdoutData = m_process->readAllStandardOutput();
    const QByteArray stderrData = m_process->readAllStandardError();
    const bool success = status == QProcess::NormalExit && exitCode == 0;

    qCDebug(lcSupportBundle) << "Support bundle process finished" << exitCode << status;
    if (!stderrData.isEmpty())
        qCDebug(lcSupportBundle) << "Support bundle stderr:" << stderrData;

    QVariantMap payload;
    const QJsonDocument doc = QJsonDocument::fromJson(stdoutData);
    if (!doc.isNull() && doc.isObject())
        payload = doc.object().toVariantMap();

    if (success) {
        m_lastResult = payload;
        const QString bundlePath = payload.value(QStringLiteral("bundle_path")).toString();
        if (!bundlePath.isEmpty())
            m_lastStatusMessage = tr("Zapisano pakiet wsparcia: %1").arg(bundlePath);
        else
            m_lastStatusMessage = tr("Zakończono eksport pakietu wsparcia.");
        if (!bundlePath.isEmpty()) {
            m_lastBundlePath = bundlePath;
            Q_EMIT lastBundlePathChanged();
        }
        Q_EMIT lastStatusMessageChanged();
        Q_EMIT lastResultChanged();
        m_lastErrorMessage.clear();
        Q_EMIT lastErrorMessageChanged();
        Q_EMIT exportFinished(true, payload);
    } else {
        QString errorMessage;
        if (!payload.isEmpty())
            errorMessage = payload.value(QStringLiteral("error")).toString();
        if (errorMessage.trimmed().isEmpty())
            errorMessage = QString::fromUtf8(stderrData).trimmed();
        if (errorMessage.trimmed().isEmpty())
            errorMessage = tr("Eksport pakietu wsparcia zakończył się błędem (kod %1)").arg(exitCode);
        m_lastErrorMessage = errorMessage;
        m_lastStatusMessage = errorMessage;
        m_lastResult = payload;
        Q_EMIT lastErrorMessageChanged();
        Q_EMIT lastStatusMessageChanged();
        Q_EMIT lastResultChanged();
        Q_EMIT exportFinished(false, payload);
    }

    resetActiveProcess();
    m_busy = false;
    Q_EMIT busyChanged();
}

void SupportBundleController::resetActiveProcess()
{
    if (m_process) {
        if (m_process->state() != QProcess::NotRunning)
            m_process->kill();
        m_process->disconnect(this);
        m_process.reset();
    }
}
