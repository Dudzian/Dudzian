#include "StrategyWorkbenchController.hpp"

#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QProcess>
#include <QStandardPaths>

Q_LOGGING_CATEGORY(lcStrategyWorkbench, "bot.shell.strategy.workbench")

namespace {
QVariantMap jsonToVariantMap(const QByteArray& payload, bool* ok)
{
    if (ok)
        *ok = false;
    QJsonParseError error;
    const QJsonDocument doc = QJsonDocument::fromJson(payload, &error);
    if (error.error != QJsonParseError::NoError || !doc.isObject())
        return {};
    if (ok)
        *ok = true;
    return doc.object().toVariantMap();
}

QVariantMap makeErrorResult(const QString& message)
{
    QVariantMap payload;
    payload.insert(QStringLiteral("ok"), false);
    QVariantList errors;
    const QString trimmed = message.trimmed();
    if (!trimmed.isEmpty())
        errors.append(trimmed);
    payload.insert(QStringLiteral("errors"), errors);
    return payload;
}
} // namespace

StrategyWorkbenchController::StrategyWorkbenchController(QObject* parent)
    : QObject(parent)
{
    m_reloadDebounce.setSingleShot(true);
    m_reloadDebounce.setInterval(250);

    connect(&m_reloadDebounce, &QTimer::timeout, this, &StrategyWorkbenchController::handleDebouncedReload);
    connect(&m_configWatcher,
            &QFileSystemWatcher::fileChanged,
            this,
            &StrategyWorkbenchController::handleWatchedPathChanged);
    connect(&m_configWatcher,
            &QFileSystemWatcher::directoryChanged,
            this,
            &StrategyWorkbenchController::handleWatchedPathChanged);
}

StrategyWorkbenchController::~StrategyWorkbenchController() = default;

void StrategyWorkbenchController::setConfigPath(const QString& path)
{
    const QString trimmed = path.trimmed();
    if (m_configPath == trimmed)
        return;
    m_configPath = trimmed;
    m_catalogInitialized = false;
    updateWatcherPaths();
    maybeTriggerInitialRefresh();
}

void StrategyWorkbenchController::setPythonExecutable(const QString& executable)
{
    const QString trimmed = executable.trimmed();
    if (trimmed.isEmpty())
        return;

    QString resolved = trimmed;
    const QFileInfo info(trimmed);
    if (!info.isAbsolute()) {
        const QString candidate = QStandardPaths::findExecutable(trimmed);
        if (!candidate.isEmpty())
            resolved = candidate;
    }

    if (m_pythonExecutable == resolved)
        return;

    m_pythonExecutable = resolved;
    m_catalogInitialized = false;
    updateWatcherPaths();
    maybeTriggerInitialRefresh();
}

void StrategyWorkbenchController::setScriptPath(const QString& path)
{
    const QString trimmed = path.trimmed();
    if (m_scriptPath == trimmed)
        return;
    m_scriptPath = trimmed;
    m_catalogInitialized = false;
    updateWatcherPaths();
    maybeTriggerInitialRefresh();
}

void StrategyWorkbenchController::clearLastError()
{
    if (m_lastError.isEmpty())
        return;
    m_lastError.clear();
    Q_EMIT lastErrorChanged();
}

bool StrategyWorkbenchController::refreshCatalog()
{
    m_pendingRefresh = false;
    if (!ensureReady()) {
        setLastError(tr("Mostek katalogu strategii nie jest poprawnie skonfigurowany."));
        return false;
    }

    m_busy = true;
    Q_EMIT busyChanged();

    const BridgeResult result = invokeBridge({QStringLiteral("--describe-catalog")});

    m_busy = false;
    Q_EMIT busyChanged();

    if (!result.ok) {
        setLastError(result.errorMessage);
        return false;
    }

    if (!parseCatalogPayload(result.stdoutData)) {
        setLastError(tr("Nie udało się sparsować danych katalogu strategii."));
        return false;
    }

    m_catalogInitialized = true;
    if (!m_lastError.isEmpty())
        clearLastError();

    Q_EMIT catalogChanged();
    return true;
}

QVariantMap StrategyWorkbenchController::validatePreset(const QVariantMap& presetPayload)
{
    QVariantMap result;
    if (!ensureReady()) {
        const QString message = tr("Mostek katalogu strategii nie jest poprawnie skonfigurowany.");
        setLastError(message);
        result = makeErrorResult(message);
        m_lastValidation = result;
        Q_EMIT lastValidationChanged();
        return result;
    }

    QJsonDocument requestDoc = QJsonDocument::fromVariant(presetPayload);

    m_busy = true;
    Q_EMIT busyChanged();

    const BridgeResult bridgeResult = invokeBridge(
        {QStringLiteral("--preset-wizard"), QStringLiteral("--wizard-mode"), QStringLiteral("validate")},
        requestDoc.toJson(QJsonDocument::Compact));

    m_busy = false;
    Q_EMIT busyChanged();

    if (!bridgeResult.ok) {
        setLastError(bridgeResult.errorMessage);
        result = makeErrorResult(bridgeResult.errorMessage);
        m_lastValidation = result;
        Q_EMIT lastValidationChanged();
        return result;
    }

    bool ok = false;
    result = jsonToVariantMap(bridgeResult.stdoutData, &ok);
    if (!ok) {
        setLastError(tr("Nie udało się zinterpretować odpowiedzi kreatora presetów."));
        result = makeErrorResult(tr("Nie udało się zinterpretować odpowiedzi kreatora presetów."));
        m_lastValidation = result;
        Q_EMIT lastValidationChanged();
        return result;
    }

    m_lastValidation = result;
    Q_EMIT lastValidationChanged();

    if (!m_lastError.isEmpty())
        clearLastError();

    return result;
}

void StrategyWorkbenchController::savePresetSnapshot(const QVariantMap& snapshot)
{
    if (snapshot.isEmpty())
        return;
    m_presetHistory.append(snapshot);
    constexpr int kMaxHistory = 20;
    while (m_presetHistory.size() > kMaxHistory)
        m_presetHistory.removeFirst();
    Q_EMIT presetHistoryChanged();
}

QVariantMap StrategyWorkbenchController::restorePreviousPreset()
{
    QVariantMap result;
    if (m_presetHistory.size() < 2) {
        result.insert(QStringLiteral("ok"), false);
        return result;
    }

    m_presetHistory.removeLast();
    const QVariantMap snapshot = m_presetHistory.last().toMap();
    result.insert(QStringLiteral("ok"), true);
    result.insert(QStringLiteral("snapshot"), snapshot);
    if (snapshot.contains(QStringLiteral("preset")))
        result.insert(QStringLiteral("preset"), snapshot.value(QStringLiteral("preset")));
    if (snapshot.contains(QStringLiteral("regime_map")))
        result.insert(QStringLiteral("regime_map"), snapshot.value(QStringLiteral("regime_map")));

    m_lastValidation = snapshot;
    Q_EMIT lastValidationChanged();

    Q_EMIT presetHistoryChanged();
    return result;
}

bool StrategyWorkbenchController::ensureReady() const
{
    if (m_configPath.isEmpty()) {
        qCWarning(lcStrategyWorkbench) << "Brak ścieżki pliku core.yaml";
        return false;
    }
    if (m_scriptPath.isEmpty()) {
        qCWarning(lcStrategyWorkbench) << "Brak ścieżki do scripts/ui_config_bridge.py";
        return false;
    }
    return true;
}

StrategyWorkbenchController::BridgeResult StrategyWorkbenchController::invokeBridge(const QStringList& args,
                                                                                     const QByteArray& stdinData) const
{
    BridgeResult result;

    QStringList fullArgs;
    fullArgs << m_scriptPath;
    fullArgs << QStringLiteral("--config") << m_configPath;
    fullArgs << args;

    QProcess process;
    process.setProcessChannelMode(QProcess::SeparateChannels);
    process.start(m_pythonExecutable, fullArgs);
    if (!process.waitForStarted()) {
        result.ok = false;
        result.errorMessage = tr("Nie udało się uruchomić mostka katalogu strategii (%1)")
                                  .arg(process.errorString());
        return result;
    }

    if (!stdinData.isEmpty())
        process.write(stdinData);
    process.closeWriteChannel();

    if (!process.waitForFinished(-1)) {
        result.ok = false;
        result.errorMessage = tr("Mostek katalogu strategii nie zakończył się poprawnie (%1)")
                                  .arg(process.errorString());
        process.kill();
        process.waitForFinished();
        return result;
    }

    result.stdoutData = process.readAllStandardOutput();
    const QByteArray stderrData = process.readAllStandardError();
    if (!stderrData.isEmpty())
        qCWarning(lcStrategyWorkbench) << "ui_config_bridge stderr:" << stderrData;

    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        result.ok = false;
        const QByteArray errorPayload = stderrData.isEmpty() ? result.stdoutData : stderrData;
        result.errorMessage = QString::fromUtf8(errorPayload).trimmed();
        if (result.errorMessage.isEmpty())
            result.errorMessage = tr("Mostek katalogu strategii zwrócił kod %1")
                                      .arg(process.exitCode());
        return result;
    }

    result.ok = true;
    return result;
}

bool StrategyWorkbenchController::parseCatalogPayload(const QByteArray& payload)
{
    bool ok = false;
    const QVariantMap map = jsonToVariantMap(payload, &ok);
    if (!ok)
        return false;

    const QVariant engines = map.value(QStringLiteral("engines"));
    if (engines.canConvert<QVariantList>())
        m_catalogEngines = engines.toList();
    else
        m_catalogEngines.clear();

    const QVariant definitions = map.value(QStringLiteral("definitions"));
    if (definitions.canConvert<QVariantList>())
        m_catalogDefinitions = definitions.toList();
    else
        m_catalogDefinitions.clear();

    const QVariant metadata = map.value(QStringLiteral("metadata"));
    if (metadata.canConvert<QVariantMap>())
        m_catalogMetadata = metadata.toMap();
    else
        m_catalogMetadata.clear();

    const QVariant blocked = map.value(QStringLiteral("blocked"));
    if (blocked.canConvert<QVariantMap>())
        m_catalogBlockedDefinitions = blocked.toMap();
    else
        m_catalogBlockedDefinitions.clear();

    const QVariant regimes = map.value(QStringLiteral("regime_templates"));
    if (regimes.canConvert<QVariantMap>())
        m_catalogRegimeTemplates = regimes.toMap();
    else
        m_catalogRegimeTemplates.clear();

    return true;
}

void StrategyWorkbenchController::setLastError(const QString& message)
{
    const QString trimmed = message.trimmed();
    if (m_lastError == trimmed)
        return;
    m_lastError = trimmed;
    Q_EMIT lastErrorChanged();
}

void StrategyWorkbenchController::updateWatcherPaths()
{
    clearWatcherPaths();

    auto addWatchTargets = [this](const QString& rawPath, bool allowDirectoryWhenMissing) {
        if (rawPath.isEmpty())
            return;

        const QFileInfo info(rawPath);
        if (info.exists() && info.isFile()) {
            const QString filePath = info.absoluteFilePath();
            if (!m_configWatcher.files().contains(filePath))
                m_configWatcher.addPath(filePath);
        }

        if (allowDirectoryWhenMissing || info.exists()) {
            const QString dirPath = info.absoluteDir().absolutePath();
            if (!dirPath.isEmpty() && !m_configWatcher.directories().contains(dirPath))
                m_configWatcher.addPath(dirPath);
        }
    };

    addWatchTargets(m_configPath, true);
    addWatchTargets(m_scriptPath, true);

    const QFileInfo pythonInfo(m_pythonExecutable);
    if (pythonInfo.isAbsolute())
        addWatchTargets(pythonInfo.absoluteFilePath(), true);
}

void StrategyWorkbenchController::clearWatcherPaths()
{
    const QStringList files = m_configWatcher.files();
    if (!files.isEmpty())
        m_configWatcher.removePaths(files);
    const QStringList dirs = m_configWatcher.directories();
    if (!dirs.isEmpty())
        m_configWatcher.removePaths(dirs);
}

void StrategyWorkbenchController::handleWatchedPathChanged(const QString& path)
{
    Q_UNUSED(path);
    updateWatcherPaths();
    scheduleAutoRefresh();
}

void StrategyWorkbenchController::scheduleAutoRefresh()
{
    m_pendingRefresh = true;
    if (m_reloadDebounce.isActive())
        m_reloadDebounce.stop();
    m_reloadDebounce.start();
}

void StrategyWorkbenchController::handleDebouncedReload()
{
    if (!m_pendingRefresh)
        return;
    if (m_busy) {
        m_reloadDebounce.start();
        return;
    }
    m_pendingRefresh = false;
    refreshCatalog();
}

void StrategyWorkbenchController::maybeTriggerInitialRefresh()
{
    if (m_catalogInitialized)
        return;
    if (m_configPath.isEmpty() || m_scriptPath.isEmpty())
        return;
    scheduleAutoRefresh();
}

