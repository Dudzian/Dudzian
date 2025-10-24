#include "StrategyWorkbenchController.hpp"

#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QProcess>
#include <QStandardPaths>
#include <QStringList>
#include <QtGlobal>

#include "utils/PathUtils.hpp"

Q_LOGGING_CATEGORY(lcStrategyWorkbench, "bot.shell.strategy.workbench")

namespace {

bool isLikelyFilesystemPath(const QString& text)
{
    if (text.isEmpty())
        return false;

    if (text.startsWith(QStringLiteral("file:"), Qt::CaseInsensitive))
        return true;

    if (text.startsWith(QStringLiteral("\\\\")))
        return true;

    if (text.startsWith(QLatin1Char('~')) || text.startsWith(QLatin1Char('.')))
        return true;

    if (text.size() >= 2 && text.at(1) == QLatin1Char(':') && text.at(0).isLetter())
        return true;

    for (const QChar ch : text) {
        if (ch == QLatin1Char('/') || ch == QLatin1Char('\'))
            return true;
    }

    return false;
}

QString normalizedFilePath(const QString& path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
        return {};
    return bot::shell::utils::expandPath(trimmed);
}

QString normalizedExecutablePath(const QString& executable)
{
    const QString trimmed = executable.trimmed();
    if (trimmed.isEmpty())
        return {};

    const QString expanded = bot::shell::utils::expandEnvironmentPlaceholders(trimmed);
    if (expanded.isEmpty())
        return {};

    if (isLikelyFilesystemPath(expanded))
        return bot::shell::utils::expandPath(expanded);

    return expanded;
}

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
    const QString normalized = normalizedFilePath(trimmed);
    const QString effective = normalized.isEmpty() ? trimmed : normalized;
    if (m_configPath == effective)
        return;
    m_configPath = effective;
    m_catalogInitialized = false;
    updateWatcherPaths();
    maybeTriggerInitialRefresh();
}

void StrategyWorkbenchController::setPythonExecutable(const QString& executable)
{
    const QString trimmed = executable.trimmed();
    if (trimmed.isEmpty())
        return;

    QString resolved = normalizedExecutablePath(trimmed);
    if (resolved.isEmpty())
        return;

    const QFileInfo info(resolved);
    if (!info.isAbsolute()) {
        const QString candidate = QStandardPaths::findExecutable(resolved);
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
    const QString normalized = normalizedFilePath(trimmed);
    const QString effective = normalized.isEmpty() ? trimmed : normalized;
    if (m_scriptPath == effective)
        return;
    m_scriptPath = effective;
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
    QString readinessError;
    if (!ensureReady(&readinessError)) {
        setLastError(readinessError.isEmpty()
                          ? tr("Mostek katalogu strategii nie jest poprawnie skonfigurowany.")
                          : readinessError);
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
    QString readinessError;
    if (!ensureReady(&readinessError)) {
        const QString message = readinessError.isEmpty()
            ? tr("Mostek katalogu strategii nie jest poprawnie skonfigurowany.")
            : readinessError;
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

bool StrategyWorkbenchController::ensureReady(QString* errorMessage) const
{
    auto setError = [errorMessage](const QString& message) {
        if (errorMessage)
            *errorMessage = message;
    };

    if (m_configPath.isEmpty()) {
        qCWarning(lcStrategyWorkbench) << "Brak ścieżki pliku core.yaml";
        setError(tr("Nie ustawiono ścieżki konfiguracji strategii."));
        return false;
    }

    const QFileInfo configInfo(m_configPath);
    if (!configInfo.exists() || !configInfo.isFile()) {
        qCWarning(lcStrategyWorkbench) << "Plik core.yaml nie istnieje" << m_configPath;
        setError(tr("Plik konfiguracji strategii nie istnieje: %1").arg(m_configPath));
        return false;
    }

    if (!configInfo.isReadable()) {
        qCWarning(lcStrategyWorkbench) << "Plik core.yaml nie posiada uprawnień do odczytu" << m_configPath;
        setError(tr("Plik konfiguracji strategii nie posiada uprawnień do odczytu: %1").arg(m_configPath));
        return false;
    }

    if (m_scriptPath.isEmpty()) {
        qCWarning(lcStrategyWorkbench) << "Brak ścieżki do scripts/ui_config_bridge.py";
        setError(tr("Nie ustawiono ścieżki mostka konfiguracji strategii."));
        return false;
    }

    const QFileInfo scriptInfo(m_scriptPath);
    if (!scriptInfo.exists() || !scriptInfo.isFile()) {
        qCWarning(lcStrategyWorkbench) << "Plik mostka nie istnieje" << m_scriptPath;
        setError(tr("Plik mostka konfiguracji strategii nie istnieje: %1").arg(m_scriptPath));
        return false;
    }

    if (!scriptInfo.isReadable()) {
        qCWarning(lcStrategyWorkbench) << "Plik mostka nie posiada uprawnień do odczytu" << m_scriptPath;
        setError(tr("Plik mostka konfiguracji strategii nie posiada uprawnień do odczytu: %1").arg(m_scriptPath));
        return false;
    }

    if (m_pythonExecutable.isEmpty()) {
        qCWarning(lcStrategyWorkbench) << "Brak interpretera Pythona";
        setError(tr("Nie ustawiono interpretera Pythona."));
        return false;
    }

    const QFileInfo pythonInfo(m_pythonExecutable);
    if (pythonInfo.isAbsolute()) {
        if (!pythonInfo.exists() || !pythonInfo.isFile()) {
            qCWarning(lcStrategyWorkbench) << "Interpreter Pythona nie istnieje" << m_pythonExecutable;
            setError(tr("Interpreter Pythona nie jest dostępny: %1").arg(m_pythonExecutable));
            return false;
        }
        if (!pythonInfo.isReadable()) {
            qCWarning(lcStrategyWorkbench) << "Interpreter Pythona nie posiada uprawnień do odczytu"
                                           << m_pythonExecutable;
            setError(tr("Interpreter Pythona nie posiada uprawnień do odczytu: %1").arg(m_pythonExecutable));
            return false;
        }
        if (!(pythonInfo.isExecutable()
              || pythonInfo.suffix().compare(QStringLiteral("exe"), Qt::CaseInsensitive) == 0)) {
            qCWarning(lcStrategyWorkbench) << "Interpreter Pythona nie jest wykonywalny" << m_pythonExecutable;
            setError(tr("Interpreter Pythona nie posiada uprawnień do uruchomienia: %1").arg(m_pythonExecutable));
            return false;
        }
    } else {
        const QString resolved = QStandardPaths::findExecutable(m_pythonExecutable);
        if (resolved.isEmpty()) {
            qCWarning(lcStrategyWorkbench)
                << "Interpreter Pythona nie został odnaleziony w PATH" << m_pythonExecutable;
            setError(tr("Interpreter Pythona nie został odnaleziony w PATH: %1").arg(m_pythonExecutable));
            return false;
        }
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

    auto addFileWatch = [this](const QString& path) {
        if (path.isEmpty() || m_configWatcher.files().contains(path))
            return;
        m_configWatcher.addPath(path);
    };

    auto addDirectoryWatch = [this](const QString& path) {
        if (path.isEmpty() || m_configWatcher.directories().contains(path))
            return;
        m_configWatcher.addPath(path);
    };

    auto registerWatchTargets = [&](const QString& rawPath) {
        if (rawPath.isEmpty())
            return;

        const QFileInfo info(rawPath);
        if (info.exists() && info.isFile()) {
            const QString absoluteFile = info.absoluteFilePath();
            addFileWatch(absoluteFile);

            const QString canonicalFile = info.canonicalFilePath();
            if (!canonicalFile.isEmpty())
                addFileWatch(canonicalFile);
        }

        const QString absoluteDir = info.absoluteDir().absolutePath();
        const QStringList watchDirs = bot::shell::utils::watchableDirectories(absoluteDir);
        for (const QString& dir : watchDirs)
            addDirectoryWatch(dir);

        if (info.exists()) {
            const QString canonicalFile = info.canonicalFilePath();
            if (!canonicalFile.isEmpty()) {
                const QFileInfo canonicalInfo(canonicalFile);
                const QString canonicalDir = canonicalInfo.absoluteDir().absolutePath();
                const QStringList canonicalDirs =
                    bot::shell::utils::watchableDirectories(canonicalDir);
                for (const QString& dir : canonicalDirs)
                    addDirectoryWatch(dir);
            }
        }
    };

    registerWatchTargets(m_configPath);
    registerWatchTargets(m_scriptPath);

    const QFileInfo pythonInfo(m_pythonExecutable);
    if (pythonInfo.isAbsolute())
        registerWatchTargets(pythonInfo.absoluteFilePath());
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

