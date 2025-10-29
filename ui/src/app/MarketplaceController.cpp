#include "MarketplaceController.hpp"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>

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

MarketplaceController::MarketplaceController(QObject* parent)
    : QObject(parent)
{
}

MarketplaceController::~MarketplaceController() = default;

void MarketplaceController::setPythonExecutable(const QString& executable)
{
    const QString trimmed = executable.trimmed();
    if (trimmed.isEmpty() || trimmed == m_pythonExecutable)
        return;
    m_pythonExecutable = trimmed;
}

void MarketplaceController::setBridgeScriptPath(const QString& path)
{
    const QString normalized = normalizePath(path);
    if (normalized == m_bridgeScriptPath)
        return;
    m_bridgeScriptPath = normalized;
}

void MarketplaceController::setPresetsDirectory(const QString& path)
{
    const QString normalized = normalizePath(path);
    if (normalized == m_presetsDir)
        return;
    m_presetsDir = normalized;
}

void MarketplaceController::setLicensesPath(const QString& path)
{
    const QString normalized = normalizePath(path);
    if (normalized == m_licensesPath)
        return;
    m_licensesPath = normalized;
}

void MarketplaceController::setSigningKeys(const QStringList& keys)
{
    QStringList sanitized;
    sanitized.reserve(keys.size());
    for (const QString& key : keys) {
        const QString trimmed = key.trimmed();
        if (trimmed.isEmpty())
            continue;
        if (!sanitized.contains(trimmed))
            sanitized.append(trimmed);
    }

    if (sanitized == m_signingKeys)
        return;
    m_signingKeys = sanitized;
}

void MarketplaceController::setSigningKeyFiles(const QStringList& paths)
{
    QStringList normalized;
    normalized.reserve(paths.size());
    for (const QString& path : paths) {
        const QString cleaned = normalizePath(path);
        if (cleaned.isEmpty())
            continue;
        if (!normalized.contains(cleaned))
            normalized.append(cleaned);
    }

    if (normalized == m_signingKeyFiles)
        return;
    m_signingKeyFiles = normalized;
}

void MarketplaceController::setFingerprintOverride(const QString& fingerprint)
{
    if (fingerprint == m_fingerprintOverride)
        return;
    m_fingerprintOverride = fingerprint;
}

MarketplaceController::BridgeResult MarketplaceController::runBridge(
    const QStringList& arguments,
    const QByteArray& stdinData)
{
    BridgeResult result;

    if (!ensureReady(&result.errorMessage))
        return result;

    QProcess process;
    QStringList fullArguments;
    fullArguments << m_bridgeScriptPath;
    fullArguments << arguments;

    process.setProgram(m_pythonExecutable);
    process.setArguments(fullArguments);
    process.start();

    if (!process.waitForStarted()) {
        result.errorMessage = tr("Nie udało się uruchomić mostka marketplace (%1)").arg(process.errorString());
        return result;
    }

    if (!stdinData.isEmpty()) {
        process.write(stdinData);
    }
    process.closeWriteChannel();

    if (!process.waitForFinished(-1)) {
        result.errorMessage = tr("Mostek marketplace nie zakończył pracy: %1").arg(process.errorString());
        process.kill();
        return result;
    }

    const QProcess::ExitStatus exitStatus = process.exitStatus();
    result.stdoutData = process.readAllStandardOutput();
    if (exitStatus != QProcess::NormalExit) {
        QString message = QString::fromUtf8(process.readAllStandardError()).trimmed();
        if (message.isEmpty())
            message = process.errorString();
        if (message.trimmed().isEmpty())
            message = tr("Mostek marketplace zakończył działanie w sposób nieoczekiwany.");
        result.errorMessage = message;
        return result;
    }

    const int exitCode = process.exitCode();
    if (exitCode != 0) {
        QString message = QString::fromUtf8(process.readAllStandardError()).trimmed();
        if (message.isEmpty())
            message = process.errorString();
        if (message.trimmed().isEmpty())
            message = tr("Mostek marketplace zakończył się kodem %1.").arg(exitCode);
        result.errorMessage = message;
        return result;
    }

    result.ok = true;
    return result;
}

bool MarketplaceController::ensureReady(QString* message) const
{
    if (m_bridgeScriptPath.isEmpty()) {
        if (message)
            *message = tr("Ścieżka do mostka marketplace nie została skonfigurowana.");
        return false;
    }
    if (m_presetsDir.isEmpty()) {
        if (message)
            *message = tr("Nie wskazano katalogu presetów marketplace.");
        return false;
    }
    if (m_licensesPath.isEmpty()) {
        if (message)
            *message = tr("Nie wskazano ścieżki stanu licencji marketplace.");
        return false;
    }
    return true;
}

QVariantList MarketplaceController::parsePresets(const QByteArray& payload) const
{
    const QJsonDocument document = QJsonDocument::fromJson(payload);
    if (!document.isObject())
        return {};
    const QJsonObject root = document.object();
    const QJsonArray presetsArray = root.value(QStringLiteral("presets")).toArray();
    QVariantList items;
    items.reserve(presetsArray.size());
    for (const QJsonValue& value : presetsArray)
        items.append(toVariant(value));
    return items;
}

QStringList MarketplaceController::buildCommonArguments() const
{
    QStringList args;
    if (!m_presetsDir.isEmpty())
        args << QStringLiteral("--presets-dir=%1").arg(m_presetsDir);
    if (!m_licensesPath.isEmpty())
        args << QStringLiteral("--licenses-path=%1").arg(m_licensesPath);
    if (!m_fingerprintOverride.isEmpty())
        args << QStringLiteral("--fingerprint") << m_fingerprintOverride;
    for (const QString& key : m_signingKeys) {
        if (!key.trimmed().isEmpty())
            args << QStringLiteral("--signing-key=%1").arg(key.trimmed());
    }
    for (const QString& file : m_signingKeyFiles) {
        if (!file.trimmed().isEmpty())
            args << QStringLiteral("--signing-key-file=%1").arg(file.trimmed());
    }
    return args;
}

bool MarketplaceController::applyPresetUpdate(const QVariantMap& presetPayload)
{
    const QString presetId = presetPayload.value(QStringLiteral("preset_id")).toString();
    if (presetId.isEmpty())
        return false;

    for (int i = 0; i < m_presets.size(); ++i) {
        QVariantMap current = m_presets.at(i).toMap();
        if (current.value(QStringLiteral("preset_id")).toString() == presetId) {
            m_presets[i] = presetPayload;
            Q_EMIT presetsChanged();
            return true;
        }
    }

    m_presets.append(presetPayload);
    Q_EMIT presetsChanged();
    return true;
}

bool MarketplaceController::refreshPresets()
{
    QString message;
    if (!ensureReady(&message)) {
        m_lastError = message;
        Q_EMIT lastErrorChanged();
        return false;
    }

    if (!m_busy) {
        m_busy = true;
        Q_EMIT busyChanged();
    }

    const BridgeResult result = runBridge(buildCommonArguments() << QStringLiteral("list"));

    m_busy = false;
    Q_EMIT busyChanged();

    if (!result.ok) {
        m_lastError = result.errorMessage;
        Q_EMIT lastErrorChanged();
        return false;
    }

    QVariantList parsed = parsePresets(result.stdoutData);
    m_presets = parsed;
    Q_EMIT presetsChanged();

    if (!m_lastError.isEmpty()) {
        m_lastError.clear();
        Q_EMIT lastErrorChanged();
    }

    return true;
}

bool MarketplaceController::activatePreset(const QString& presetId, const QVariantMap& licensePayload)
{
    QString message;
    if (!ensureReady(&message)) {
        m_lastError = message;
        Q_EMIT lastErrorChanged();
        return false;
    }
    if (presetId.trimmed().isEmpty()) {
        m_lastError = tr("Identyfikator presetu jest wymagany.");
        Q_EMIT lastErrorChanged();
        return false;
    }

    QJsonDocument doc(QJsonObject::fromVariantMap(licensePayload));
    QByteArray stdinData = doc.toJson(QJsonDocument::Compact);

    if (!m_busy) {
        m_busy = true;
        Q_EMIT busyChanged();
    }

    QStringList args = buildCommonArguments();
    args << QStringLiteral("activate");
    args << QStringLiteral("--preset-id");
    args << presetId;

    const BridgeResult result = runBridge(args, stdinData);

    m_busy = false;
    Q_EMIT busyChanged();

    if (!result.ok) {
        m_lastError = result.errorMessage;
        Q_EMIT lastErrorChanged();
        return false;
    }

    const QJsonDocument document = QJsonDocument::fromJson(result.stdoutData);
    const QJsonObject root = document.object();
    const QVariantMap preset = root.value(QStringLiteral("preset")).toObject().toVariantMap();
    if (!preset.isEmpty())
        applyPresetUpdate(preset);

    if (!m_lastError.isEmpty()) {
        m_lastError.clear();
        Q_EMIT lastErrorChanged();
    }

    return true;
}

bool MarketplaceController::deactivatePreset(const QString& presetId)
{
    QString message;
    if (!ensureReady(&message)) {
        m_lastError = message;
        Q_EMIT lastErrorChanged();
        return false;
    }
    if (presetId.trimmed().isEmpty()) {
        m_lastError = tr("Identyfikator presetu jest wymagany.");
        Q_EMIT lastErrorChanged();
        return false;
    }

    if (!m_busy) {
        m_busy = true;
        Q_EMIT busyChanged();
    }

    QStringList args = buildCommonArguments();
    args << QStringLiteral("deactivate");
    args << QStringLiteral("--preset-id");
    args << presetId;

    const BridgeResult result = runBridge(args);

    m_busy = false;
    Q_EMIT busyChanged();

    if (!result.ok) {
        m_lastError = result.errorMessage;
        Q_EMIT lastErrorChanged();
        return false;
    }

    const QJsonDocument document = QJsonDocument::fromJson(result.stdoutData);
    const QJsonObject root = document.object();
    const QVariantMap preset = root.value(QStringLiteral("preset")).toObject().toVariantMap();
    if (!preset.isEmpty())
        applyPresetUpdate(preset);

    if (!m_lastError.isEmpty()) {
        m_lastError.clear();
        Q_EMIT lastErrorChanged();
    }

    return true;
}
