#include "PortfolioManagerController.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QVariantMap>

namespace {
QString normalizePath(const QString& path)
{
    if (path.trimmed().isEmpty())
        return {};
    QFileInfo info(path);
    if (info.isRelative())
        info.setFile(QDir::current().absoluteFilePath(path));
    return QDir::cleanPath(info.absoluteFilePath());
}

QVariant toVariant(const QJsonValue& value)
{
    if (value.isObject())
        return value.toObject().toVariantMap();
    if (value.isArray())
        return value.toArray().toVariantList();
    return value.toVariant();
}
} // namespace

PortfolioManagerController::PortfolioManagerController(QObject* parent)
    : QObject(parent)
{
    m_portfolioDecisionLogPath = normalizePath(QStringLiteral("logs/portfolio_decisions.jsonl"));
}

PortfolioManagerController::~PortfolioManagerController() = default;

void PortfolioManagerController::setPythonExecutable(const QString& executable)
{
    const QString trimmed = executable.trimmed();
    if (trimmed.isEmpty() || trimmed == m_pythonExecutable)
        return;
    m_pythonExecutable = trimmed;
}

void PortfolioManagerController::setBridgeScriptPath(const QString& path)
{
    const QString normalized = normalizePath(path);
    if (normalized == m_bridgeScriptPath)
        return;
    m_bridgeScriptPath = normalized;
}

void PortfolioManagerController::setStorePath(const QString& path)
{
    const QString normalized = normalizePath(path);
    if (normalized == m_storePath)
        return;
    m_storePath = normalized;
}

void PortfolioManagerController::setPortfolioDecisionLogPath(const QString& path)
{
    const QString normalized = normalizePath(path);
    if (normalized == m_portfolioDecisionLogPath)
        return;
    m_portfolioDecisionLogPath = normalized;
}

PortfolioManagerController::BridgeResult PortfolioManagerController::runBridge(const QStringList& arguments, const QByteArray& stdinData)
{
    BridgeResult result;
    if (!ensureReady(&result.errorMessage))
        return result;

    QProcess process;
    QStringList fullArguments;
    fullArguments << m_bridgeScriptPath;
    fullArguments << QStringLiteral("--store");
    fullArguments << m_storePath;
    fullArguments << arguments;

    process.setProgram(m_pythonExecutable);
    process.setArguments(fullArguments);
    process.start();

    if (!process.waitForStarted()) {
        result.errorMessage = tr("Nie udało się uruchomić mostka portfelowego (%1)").arg(process.errorString());
        return result;
    }

    if (!stdinData.isEmpty())
        process.write(stdinData);
    process.closeWriteChannel();

    if (!process.waitForFinished(-1)) {
        result.errorMessage = tr("Mostek portfelowy nie zakończył pracy: %1").arg(process.errorString());
        process.kill();
        return result;
    }

    result.stdoutData = process.readAllStandardOutput();

    if (process.exitStatus() != QProcess::NormalExit) {
        QString message = QString::fromUtf8(process.readAllStandardError()).trimmed();
        if (message.isEmpty())
            message = process.errorString();
        if (message.trimmed().isEmpty())
            message = tr("Mostek portfelowy zakończył działanie w sposób nieoczekiwany.");
        result.errorMessage = message;
        return result;
    }

    const int exitCode = process.exitCode();
    if (exitCode != 0) {
        QString message = QString::fromUtf8(process.readAllStandardError()).trimmed();
        if (message.isEmpty())
            message = process.errorString();
        if (message.trimmed().isEmpty())
            message = tr("Mostek portfelowy zakończył się kodem %1.").arg(exitCode);
        result.errorMessage = message;
        return result;
    }

    result.ok = true;
    return result;
}

bool PortfolioManagerController::ensureReady(QString* message) const
{
    if (m_bridgeScriptPath.isEmpty()) {
        if (message)
            *message = tr("Ścieżka do mostka portfelowego nie została skonfigurowana.");
        return false;
    }
    if (m_storePath.isEmpty()) {
        if (message)
            *message = tr("Nie wskazano ścieżki magazynu konfiguracji portfeli.");
        return false;
    }
    return true;
}

QVariantList PortfolioManagerController::parsePortfolios(const QByteArray& payload) const
{
    const QJsonDocument document = QJsonDocument::fromJson(payload);
    if (!document.isObject())
        return {};
    const QJsonObject root = document.object();
    const QJsonArray array = root.value(QStringLiteral("portfolios")).toArray();
    QVariantList items;
    items.reserve(array.size());
    for (const QJsonValue& value : array)
        items.append(toVariant(value));
    return items;
}

bool PortfolioManagerController::refreshPortfolios()
{
    if (m_busy)
        return false;

    m_busy = true;
    emit busyChanged();

    BridgeResult result = runBridge(QStringList{QStringLiteral("list")});
    if (!result.ok) {
        m_lastError = result.errorMessage;
        emit lastErrorChanged();
        m_busy = false;
        emit busyChanged();
        return false;
    }

    m_portfolios = parsePortfolios(result.stdoutData);
    emit portfoliosChanged();
    if (!m_lastError.isEmpty()) {
        m_lastError.clear();
        emit lastErrorChanged();
    }
    m_busy = false;
    emit busyChanged();
    return true;
}

bool PortfolioManagerController::applyPortfolio(const QVariantMap& payload)
{
    if (m_busy)
        return false;

    QJsonDocument document = QJsonDocument::fromVariant(payload);
    QByteArray stdinPayload = document.toJson(QJsonDocument::Compact);

    m_busy = true;
    emit busyChanged();

    BridgeResult result = runBridge(QStringList{QStringLiteral("apply")}, stdinPayload);
    if (!result.ok) {
        m_lastError = result.errorMessage;
        emit lastErrorChanged();
        m_busy = false;
        emit busyChanged();
        return false;
    }

    m_busy = false;
    emit busyChanged();
    return refreshPortfolios();
}

bool PortfolioManagerController::refreshGovernorDecisions(int limit)
{
    if (m_busy)
        return false;

    if (limit <= 0)
        limit = 10;

    m_busy = true;
    emit busyChanged();

    if (m_portfolioDecisionLogPath.isEmpty()) {
        m_busy = false;
        emit busyChanged();
        if (!m_governorDecisions.isEmpty()) {
            m_governorDecisions.clear();
            emit governorDecisionsChanged();
        }
        return true;
    }

    QFile file(m_portfolioDecisionLogPath);
    if (!file.exists()) {
        m_busy = false;
        emit busyChanged();
        if (!m_governorDecisions.isEmpty()) {
            m_governorDecisions.clear();
            emit governorDecisionsChanged();
        }
        return true;
    }

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        m_busy = false;
        emit busyChanged();
        const QString message = tr("Nie udało się otworzyć dziennika decyzji portfela (%1)").arg(file.errorString());
        if (m_lastError != message) {
            m_lastError = message;
            emit lastErrorChanged();
        }
        return false;
    }

    QList<QByteArray> buffer;
    while (!file.atEnd()) {
        const QByteArray line = file.readLine();
        if (line.trimmed().isEmpty())
            continue;
        buffer.append(line);
        if (buffer.size() > limit)
            buffer.pop_front();
    }
    file.close();

    m_busy = false;
    emit busyChanged();

    QVariantList entries;
    entries.reserve(buffer.size());
    for (const QByteArray& raw : buffer) {
        const QJsonDocument document = QJsonDocument::fromJson(raw);
        if (!document.isObject())
            continue;
        entries.append(document.object().toVariantMap());
    }

    m_governorDecisions = entries;
    emit governorDecisionsChanged();
    if (!m_lastError.isEmpty()) {
        m_lastError.clear();
        emit lastErrorChanged();
    }
    return true;
}

bool PortfolioManagerController::removePortfolio(const QString& portfolioId)
{
    if (m_busy)
        return false;

    const QString trimmed = portfolioId.trimmed();
    if (trimmed.isEmpty()) {
        m_lastError = tr("Identyfikator portfela nie może być pusty.");
        emit lastErrorChanged();
        return false;
    }

    m_busy = true;
    emit busyChanged();

    BridgeResult result = runBridge(QStringList{QStringLiteral("remove"), trimmed});
    if (!result.ok) {
        m_lastError = result.errorMessage;
        emit lastErrorChanged();
        m_busy = false;
        emit busyChanged();
        return false;
    }

    m_busy = false;
    emit busyChanged();
    return refreshPortfolios();
}
