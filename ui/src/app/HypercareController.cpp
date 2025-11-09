#include "app/HypercareController.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QStringList>

#include "utils/PathUtils.hpp"

namespace {
Q_LOGGING_CATEGORY(lcHypercare, "bot.shell.hypercare")
}

HypercareController::HypercareController(QObject* parent)
    : QObject(parent)
{
    m_cycleScript = normalizePath(m_cycleScript);
    m_bundleSource = normalizePath(m_bundleSource);
    m_bundleOutputDir = normalizePath(m_bundleOutputDir);
    m_failoverPlanPath = normalizePath(m_failoverPlanPath);
    m_selfHealRulesPath = normalizePath(m_selfHealRulesPath);
    m_selfHealOutputPath = normalizePath(m_selfHealOutputPath);
    m_stage6SummaryPath = normalizePath(m_stage6SummaryPath);
}

HypercareController::~HypercareController()
{
    resetProcess();
}

void HypercareController::setPythonExecutable(const QString& value)
{
    const QString normalized = normalizePath(value);
    if (normalized == m_pythonExecutable)
        return;
    m_pythonExecutable = normalized.isEmpty() ? QStringLiteral("python3") : normalized;
    Q_EMIT pythonExecutableChanged();
}

void HypercareController::setCycleScript(const QString& value)
{
    const QString normalized = normalizePath(value);
    if (normalized == m_cycleScript)
        return;
    m_cycleScript = normalized;
    Q_EMIT cycleScriptChanged();
}

void HypercareController::setBundleSource(const QString& value)
{
    const QString normalized = normalizePath(value);
    if (normalized == m_bundleSource)
        return;
    m_bundleSource = normalized;
    Q_EMIT bundleSourceChanged();
}

void HypercareController::setBundleOutputDir(const QString& value)
{
    const QString normalized = normalizePath(value);
    if (normalized == m_bundleOutputDir)
        return;
    m_bundleOutputDir = normalized;
    Q_EMIT bundleOutputDirChanged();
}

void HypercareController::setFailoverPlanPath(const QString& value)
{
    const QString normalized = normalizePath(value);
    if (normalized == m_failoverPlanPath)
        return;
    m_failoverPlanPath = normalized;
    Q_EMIT failoverPlanPathChanged();
}

void HypercareController::setSelfHealRulesPath(const QString& value)
{
    const QString normalized = normalizePath(value);
    if (normalized == m_selfHealRulesPath)
        return;
    m_selfHealRulesPath = normalized;
    Q_EMIT selfHealRulesPathChanged();
}

void HypercareController::setSelfHealOutputPath(const QString& value)
{
    const QString normalized = normalizePath(value);
    if (normalized == m_selfHealOutputPath)
        return;
    m_selfHealOutputPath = normalized;
    Q_EMIT selfHealOutputPathChanged();
}

void HypercareController::setSigningKeyPath(const QString& value)
{
    const QString normalized = normalizePath(value);
    if (normalized == m_signingKeyPath)
        return;
    m_signingKeyPath = normalized;
    Q_EMIT signingKeyPathChanged();
}

void HypercareController::setSigningKeyId(const QString& value)
{
    if (value == m_signingKeyId)
        return;
    m_signingKeyId = value;
    Q_EMIT signingKeyIdChanged();
}

void HypercareController::setStage6SummaryPath(const QString& value)
{
    const QString normalized = normalizePath(value);
    if (normalized == m_stage6SummaryPath)
        return;
    m_stage6SummaryPath = normalized;
    Q_EMIT stage6SummaryPathChanged();
}

QVariantMap HypercareController::loadSummary(const QString& path) const
{
    const QString resolved = normalizePath(path.isEmpty() ? m_stage6SummaryPath : path);
    QFile file(resolved);
    if (!file.exists()) {
        qCWarning(lcHypercare) << "Summary file does not exist" << resolved;
        return {};
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCWarning(lcHypercare) << "Cannot open summary file" << resolved << file.errorString();
        return {};
    }
    const auto document = QJsonDocument::fromJson(file.readAll());
    if (!document.isObject()) {
        qCWarning(lcHypercare) << "Summary file is not a JSON object" << resolved;
        return {};
    }
    return document.object().toVariantMap();
}

bool HypercareController::triggerSelfHealing()
{
    if (m_running) {
        qCWarning(lcHypercare) << "Self-healing already running";
        return false;
    }

    QString scriptPath = m_cycleScript;
    if (scriptPath.isEmpty()) {
        qCWarning(lcHypercare) << "Brak ścieżki do skryptu run_stage6_resilience_cycle.py";
        return false;
    }

    resetProcess();
    m_process = new QProcess(this);
    connect(m_process, &QProcess::finished, this, &HypercareController::handleProcessFinished);
    connect(m_process, &QProcess::errorOccurred, this, &HypercareController::handleProcessError);
    connect(m_process, &QProcess::readyReadStandardOutput, this, &HypercareController::handleReadyRead);
    connect(m_process, &QProcess::readyReadStandardError, this, &HypercareController::handleReadyRead);

    QStringList args;
    args << scriptPath;
    args << "--source" << m_bundleSource;
    args << "--bundle-output-dir" << m_bundleOutputDir;
    args << "--plan" << m_failoverPlanPath;
    args << "--self-heal-config" << m_selfHealRulesPath;
    args << "--self-heal-output" << m_selfHealOutputPath;
    args << "--self-heal-mode" << QStringLiteral("execute");
    if (!m_signingKeyPath.isEmpty())
        args << "--signing-key-path" << m_signingKeyPath;
    if (!m_signingKeyId.isEmpty())
        args << "--signing-key-id" << m_signingKeyId;

    m_outputBuffer.clear();
    m_errorBuffer.clear();
    m_lastMessage.clear();
    m_lastError.clear();

    m_process->setProgram(m_pythonExecutable);
    m_process->setArguments(args);
    m_process->setProcessChannelMode(QProcess::SeparateChannels);
    m_process->start();
    m_running = true;
    Q_EMIT runningChanged();
    return true;
}

void HypercareController::handleProcessFinished(int exitCode, QProcess::ExitStatus)
{
    handleReadyRead();
    m_running = false;
    Q_EMIT runningChanged();

    if (!m_outputBuffer.isEmpty()) {
        const QString trimmed = m_outputBuffer.trimmed();
        if (trimmed != m_lastMessage) {
            m_lastMessage = trimmed;
            Q_EMIT lastMessageChanged();
        }
    }
    if (!m_errorBuffer.isEmpty()) {
        const QString trimmedError = m_errorBuffer.trimmed();
        if (trimmedError != m_lastError) {
            m_lastError = trimmedError;
            Q_EMIT lastErrorChanged();
        }
    }

    Q_EMIT selfHealingCompleted(exitCode);
    resetProcess();
}

void HypercareController::handleProcessError(QProcess::ProcessError error)
{
    handleReadyRead();
    m_running = false;
    Q_EMIT runningChanged();
    const QString message = QString::fromLatin1("Błąd uruchomienia self-healing: %1").arg(static_cast<int>(error));
    if (message != m_lastError) {
        m_lastError = message;
        Q_EMIT lastErrorChanged();
    }
    resetProcess();
}

void HypercareController::handleReadyRead()
{
    if (!m_process)
        return;
    const QByteArray stdoutData = m_process->readAllStandardOutput();
    if (!stdoutData.isEmpty()) {
        m_outputBuffer += QString::fromUtf8(stdoutData);
        const QString trimmed = m_outputBuffer.trimmed();
        if (trimmed != m_lastMessage) {
            m_lastMessage = trimmed;
            Q_EMIT lastMessageChanged();
        }
    }
    const QByteArray stderrData = m_process->readAllStandardError();
    if (!stderrData.isEmpty()) {
        m_errorBuffer += QString::fromUtf8(stderrData);
        const QString trimmedError = m_errorBuffer.trimmed();
        if (trimmedError != m_lastError) {
            m_lastError = trimmedError;
            Q_EMIT lastErrorChanged();
        }
    }
}

void HypercareController::resetProcess()
{
    if (!m_process)
        return;
    m_process->disconnect(this);
    if (m_process->state() != QProcess::NotRunning)
        m_process->kill();
    m_process->deleteLater();
    m_process = nullptr;
}

QString HypercareController::normalizePath(const QString& path) const
{
    if (path.trimmed().isEmpty())
        return QString();
    return bot::shell::utils::expandPath(path.trimmed());
}
