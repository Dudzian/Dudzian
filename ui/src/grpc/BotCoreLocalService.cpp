#include "BotCoreLocalService.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcessEnvironment>
#include <QStandardPaths>
#include <QVariant>
#include <QtGlobal>
#include <limits>

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
    m_streamPollTimer.setSingleShot(true);
    connect(&m_streamPollTimer, &QTimer::timeout, this, &BotCoreLocalService::sendStreamPoll);
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
    m_pendingMethods.clear();
    m_streamPollTimer.stop();
    m_streamSubscriptionId.clear();
    m_streamSymbol.clear();
    m_streamActive = false;
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
            continue;
        }
        if (obj.contains(QStringLiteral("id")))
            handleRpcMessage(obj);
    }
    return ready;
}

quint64 BotCoreLocalService::nextRequestId()
{
    if (m_nextRequestId == std::numeric_limits<quint64>::max())
        m_nextRequestId = 1;
    return m_nextRequestId++;
}

bool BotCoreLocalService::sendJsonRequest(const QJsonObject& object, quint64 id, const QString& method)
{
    if (!isRunning())
        return false;
    QJsonDocument doc(object);
    QByteArray payload = doc.toJson(QJsonDocument::Compact);
    payload.append('\n');
    const qint64 written = m_process.write(payload);
    if (written != payload.size())
        return false;
    m_process.waitForBytesWritten(50);
    m_pendingMethods.insert(id, method);
    return true;
}

void BotCoreLocalService::handleRpcMessage(const QJsonObject& object)
{
    const quint64 id = static_cast<quint64>(object.value(QStringLiteral("id")).toVariant().toULongLong());
    if (id == 0)
        return;
    const QString method = m_pendingMethods.take(id);
    if (method.isEmpty())
        return;
    if (object.contains(QStringLiteral("error"))) {
        const QJsonObject error = object.value(QStringLiteral("error")).toObject();
        const QString message = error.value(QStringLiteral("message")).toString();
        if (method == QStringLiteral("market_data.stream_ohlcv"))
            emit ohlcvStreamError(message.isEmpty() ? tr("Błąd strumienia OHLCV") : message);
        return;
    }
    const QJsonObject result = object.value(QStringLiteral("result")).toObject();
    if (method == QStringLiteral("market_data.stream_ohlcv"))
        handleStreamResponse(result);
}

void BotCoreLocalService::handleStreamResponse(const QJsonObject& result)
{
    if (!m_streamActive)
        m_streamActive = true;
    const QString subscription = result.value(QStringLiteral("subscription_id")).toString();
    if (!subscription.isEmpty())
        m_streamSubscriptionId = subscription;
    const QString effectiveId = m_streamSubscriptionId;
    if (result.value(QStringLiteral("cancelled")).toBool(false)) {
        if (!effectiveId.isEmpty())
            emit ohlcvStreamClosed(effectiveId);
        m_streamPollTimer.stop();
        m_streamSubscriptionId.clear();
        m_streamActive = false;
        return;
    }

    const QJsonArray snapshotArray = result.value(QStringLiteral("snapshot")).toArray();
    if (!snapshotArray.isEmpty() && !effectiveId.isEmpty())
        emit ohlcvSnapshotReady(snapshotArray.toVariantList(), effectiveId);

    const QJsonArray updatesArray = result.value(QStringLiteral("updates")).toArray();
    if (!updatesArray.isEmpty() && !effectiveId.isEmpty())
        emit ohlcvUpdatesReady(updatesArray.toVariantList(), effectiveId);

    const bool hasMore = result.value(QStringLiteral("has_more")).toBool(false);
    if (hasMore && !effectiveId.isEmpty()) {
        scheduleNextPoll();
        return;
    }

    if (!effectiveId.isEmpty())
        emit ohlcvStreamClosed(effectiveId);
    m_streamPollTimer.stop();
    m_streamSubscriptionId.clear();
    m_streamActive = false;
}

void BotCoreLocalService::scheduleNextPoll(int delayMs)
{
    if (!m_streamActive || m_streamSubscriptionId.isEmpty())
        return;
    if (delayMs < 0)
        delayMs = 0;
    m_streamPollTimer.start(delayMs);
}

void BotCoreLocalService::sendStreamPoll()
{
    if (!m_streamActive || m_streamSubscriptionId.isEmpty())
        return;
    QJsonObject params;
    params.insert(QStringLiteral("subscription_id"), m_streamSubscriptionId);
    params.insert(QStringLiteral("timeout_ms"), m_streamTimeoutMs);
    params.insert(QStringLiteral("max_updates"), m_streamMaxUpdates);
    const quint64 id = nextRequestId();
    QJsonObject request;
    request.insert(QStringLiteral("id"), static_cast<double>(id));
    request.insert(QStringLiteral("method"), QStringLiteral("market_data.stream_ohlcv"));
    request.insert(QStringLiteral("params"), params);
    if (!sendJsonRequest(request, id, QStringLiteral("market_data.stream_ohlcv"))) {
        emit ohlcvStreamError(tr("Nie udało się odczytać danych strumienia OHLCV"));
        m_streamPollTimer.stop();
        m_streamActive = false;
    }
}

void BotCoreLocalService::startOhlcvStream(const QString& symbol, int maxUpdates, int timeoutMs)
{
    if (!isRunning())
        return;
    const QString trimmed = symbol.trimmed();
    if (trimmed.isEmpty())
        return;
    m_streamSymbol = trimmed;
    m_streamMaxUpdates = qMax(0, maxUpdates);
    m_streamTimeoutMs = qMax(0, timeoutMs);
    m_streamSubscriptionId.clear();
    m_streamActive = true;
    QJsonObject params;
    params.insert(QStringLiteral("symbol"), trimmed);
    params.insert(QStringLiteral("continuous"), true);
    params.insert(QStringLiteral("max_updates"), m_streamMaxUpdates);
    params.insert(QStringLiteral("timeout_ms"), m_streamTimeoutMs);
    const quint64 id = nextRequestId();
    QJsonObject request;
    request.insert(QStringLiteral("id"), static_cast<double>(id));
    request.insert(QStringLiteral("method"), QStringLiteral("market_data.stream_ohlcv"));
    request.insert(QStringLiteral("params"), params);
    if (!sendJsonRequest(request, id, QStringLiteral("market_data.stream_ohlcv"))) {
        m_streamActive = false;
        emit ohlcvStreamError(tr("Nie udało się zainicjować strumienia OHLCV"));
    }
}

void BotCoreLocalService::stopOhlcvStream()
{
    if (!m_streamActive)
        return;
    m_streamActive = false;
    m_streamPollTimer.stop();
    if (m_streamSubscriptionId.isEmpty()) {
        m_streamSymbol.clear();
        return;
    }
    QJsonObject params;
    params.insert(QStringLiteral("subscription_id"), m_streamSubscriptionId);
    params.insert(QStringLiteral("cancel"), true);
    const quint64 id = nextRequestId();
    QJsonObject request;
    request.insert(QStringLiteral("id"), static_cast<double>(id));
    request.insert(QStringLiteral("method"), QStringLiteral("market_data.stream_ohlcv"));
    request.insert(QStringLiteral("params"), params);
    sendJsonRequest(request, id, QStringLiteral("market_data.stream_ohlcv"));
    m_streamSymbol.clear();
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
