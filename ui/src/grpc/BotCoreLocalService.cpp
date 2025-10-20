#include "BotCoreLocalService.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcessEnvironment>
#include <QStandardPaths>
#include <QtGlobal>

namespace {
constexpr int kReadChunkMs = 200;

QString normalizedPath(const QString& path)
{
    if (path.trimmed().isEmpty())
        return {};
    QDir dir(path);
    if (!dir.isAbsolute())
        dir = QDir(QDir::current().absoluteFilePath(path));
    return QDir::cleanPath(dir.absolutePath());
}

QString joinPythonPath(const QString& existing, const QString& extra)
{
    if (extra.isEmpty())
        return existing;
    if (existing.isEmpty())
        return extra;
#ifdef Q_OS_WIN
    constexpr QChar separator = QLatin1Char(';');
#else
    constexpr QChar separator = QLatin1Char(':');
#endif
    return extra + separator + existing;
}

} // namespace

BotCoreLocalService::BotCoreLocalService(QObject* parent)
    : QObject(parent)
{
    connect(&m_process, &QProcess::readyReadStandardOutput, this, &BotCoreLocalService::handleReadyRead);
    connect(&m_process, &QProcess::readyReadStandardError, this, &BotCoreLocalService::handleReadyReadError);
    connect(&m_process, &QProcess::finished, this, &BotCoreLocalService::handleProcessFinished);
    m_process.setProcessChannelMode(QProcess::SeparateChannels);
}

BotCoreLocalService::~BotCoreLocalService()
{
    stop();
}

void BotCoreLocalService::setPythonExecutable(const QString& executable)
{
    if (!executable.trimmed().isEmpty())
        m_pythonExecutable = executable.trimmed();
}

void BotCoreLocalService::setModuleName(const QString& moduleName)
{
    if (!moduleName.trimmed().isEmpty())
        m_moduleName = moduleName.trimmed();
}

void BotCoreLocalService::setDatasetPath(const QString& datasetPath)
{
    m_datasetPath = normalizedPath(datasetPath);
}

void BotCoreLocalService::setHost(const QString& host)
{
    if (!host.trimmed().isEmpty())
        m_host = host.trimmed();
}

void BotCoreLocalService::setPort(int port)
{
    m_port = port;
}

void BotCoreLocalService::setStreamRepeat(bool enabled)
{
    m_streamRepeat = enabled;
}

void BotCoreLocalService::setStreamInterval(double seconds)
{
    m_streamInterval = seconds;
}

void BotCoreLocalService::setRepoRoot(const QString& path)
{
    m_repoRoot = normalizedPath(path);
}

void BotCoreLocalService::resetState()
{
    m_endpoint.clear();
    m_lastError.clear();
    m_stdoutBuffer.clear();
    m_stderrBuffer.clear();
}

void BotCoreLocalService::handleReadyRead()
{
    parseStdoutBuffer();
}

void BotCoreLocalService::handleReadyReadError()
{
    appendStderr(m_process.readAllStandardError());
}

bool BotCoreLocalService::parseStdoutBuffer()
{
    m_stdoutBuffer += m_process.readAllStandardOutput();
    bool ready = false;
    while (true) {
        const int newlineIndex = m_stdoutBuffer.indexOf('\n');
        if (newlineIndex < 0)
            break;
        const QByteArray line = m_stdoutBuffer.left(newlineIndex).trimmed();
        m_stdoutBuffer.remove(0, newlineIndex + 1);
        if (line.isEmpty())
            continue;
        const QJsonDocument doc = QJsonDocument::fromJson(line);
        if (!doc.isObject())
            continue;
        const QJsonObject obj = doc.object();
        if (obj.value(QStringLiteral("event")).toString() == QStringLiteral("ready")) {
            const QString address = obj.value(QStringLiteral("address")).toString();
            if (!address.isEmpty()) {
                m_endpoint = address;
                ready = true;
                break;
            }
        }
    }
    return ready;
}

bool BotCoreLocalService::waitForReady(int timeoutMs)
{
    QElapsedTimer timer;
    timer.start();
    while (timer.elapsed() < timeoutMs) {
        if (parseStdoutBuffer())
            return true;
        const int remaining = timeoutMs - static_cast<int>(timer.elapsed());
        if (!m_process.waitForReadyRead(qMax(50, qMin(kReadChunkMs, remaining))))
            break;
        if (parseStdoutBuffer())
            return true;
    }
    parseStdoutBuffer();
    return !m_endpoint.isEmpty();
}

QString BotCoreLocalService::locateRepoRootFromCwd() const
{
    QDir dir(QDir::current());
    for (int depth = 0; depth < 12; ++depth) {
        if (dir.exists(QStringLiteral("bot_core")) && dir.exists(QStringLiteral("ui")))
            return dir.absolutePath();
        if (!dir.cdUp())
            break;
    }
    return {};
}

bool BotCoreLocalService::start(int timeoutMs)
{
    if (isRunning())
        stop();

    resetState();

    QString repoRoot = m_repoRoot;
    if (repoRoot.isEmpty())
        repoRoot = locateRepoRootFromCwd();

    QStringList args;
    args << QStringLiteral("-m") << m_moduleName;
    args << QStringLiteral("--host") << m_host;
    args << QStringLiteral("--port") << QString::number(m_port);
    if (!m_datasetPath.isEmpty())
        args << QStringLiteral("--dataset") << m_datasetPath;
    if (m_streamRepeat)
        args << QStringLiteral("--stream-repeat");
    if (m_streamInterval > 0.0)
        args << QStringLiteral("--stream-interval")
             << QString::number(m_streamInterval, 'f', 3);
    args << QStringLiteral("--log-level") << QStringLiteral("WARNING");

    m_process.setProgram(m_pythonExecutable);
    m_process.setArguments(args);

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    if (!repoRoot.isEmpty()) {
        const QString merged = joinPythonPath(env.value(QStringLiteral("PYTHONPATH")), repoRoot);
        env.insert(QStringLiteral("PYTHONPATH"), merged);
        m_process.setWorkingDirectory(repoRoot);
    }
    env.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
    m_process.setProcessEnvironment(env);

    m_process.start();
    if (!m_process.waitForStarted(timeoutMs)) {
        m_lastError = tr("Nie udało się uruchomić lokalnego serwisu bot_core (%1)")
                           .arg(m_process.errorString());
        stop();
        return false;
    }

    if (!waitForReady(timeoutMs)) {
        appendStderr(m_process.readAllStandardError());
        m_lastError = tr("Lokalny serwis bot_core nie potwierdził gotowości");
        const QString detail = lastStderrLines();
        if (!detail.isEmpty())
            m_lastError += QStringLiteral(" • %1").arg(detail);
        stop();
        return false;
    }

    return !m_endpoint.isEmpty();
}

void BotCoreLocalService::handleProcessFinished(int exitCode, QProcess::ExitStatus status)
{
    Q_UNUSED(exitCode);
    Q_UNUSED(status);
    appendStderr(m_process.readAllStandardError());
    if (m_endpoint.isEmpty() && m_lastError.isEmpty()) {
        m_lastError = tr("Proces lokalnego serwisu bot_core zakończył się przed potwierdzeniem gotowości");
        const QString detail = lastStderrLines();
        if (!detail.isEmpty())
            m_lastError += QStringLiteral(" • %1").arg(detail);
    }
}

void BotCoreLocalService::stop(int timeoutMs)
{
    if (!isRunning())
        return;
    m_process.terminate();
    if (!m_process.waitForFinished(timeoutMs)) {
        m_process.kill();
        m_process.waitForFinished(timeoutMs);
    }
    appendStderr(m_process.readAllStandardError());
}

void BotCoreLocalService::appendStderr(const QByteArray& chunk)
{
    if (chunk.isEmpty())
        return;
    m_stderrBuffer += chunk;
    constexpr int kMaxBuffer = 4096;
    if (m_stderrBuffer.size() > kMaxBuffer)
        m_stderrBuffer = m_stderrBuffer.right(kMaxBuffer);
}

QString BotCoreLocalService::lastStderrLines(int maxLines) const
{
    if (m_stderrBuffer.isEmpty() || maxLines <= 0)
        return {};

    QByteArray normalized = m_stderrBuffer;
    normalized.replace('\r', '\n');
    const QList<QByteArray> lines = normalized.split('\n');
    QStringList tail;
    tail.reserve(maxLines);
    for (int i = lines.size() - 1; i >= 0 && tail.size() < maxLines; --i) {
        const QByteArray trimmed = lines.at(i).trimmed();
        if (trimmed.isEmpty())
            continue;
        tail.prepend(QString::fromUtf8(trimmed));
    }
    return tail.join(QStringLiteral(" | "));
}
