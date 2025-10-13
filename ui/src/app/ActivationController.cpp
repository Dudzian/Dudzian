#include "ActivationController.hpp"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QProcess>
#include <QTextStream>

Q_LOGGING_CATEGORY(lcActivation, "bot.shell.activation")

namespace {

QString envValue(const char* key, const QString& fallback = QString())
{
    const QByteArray env = qgetenv(key);
    if (env.isEmpty())
        return fallback;
    return QString::fromUtf8(env).trimmed();
}

QStringList readKeyArguments(const QString& path)
{
    QFile file(path);
    if (!file.exists()) {
        qCWarning(lcActivation) << "Plik z kluczami fingerprint nie istnieje:" << path;
        return {};
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCWarning(lcActivation) << "Nie udało się otworzyć pliku z kluczami" << path << file.errorString();
        return {};
    }
    const QByteArray data = file.readAll();
    QJsonParseError error{};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &error);
    if (error.error != QJsonParseError::NoError) {
        qCWarning(lcActivation) << "Nie udało się sparsować pliku z kluczami" << path << error.errorString();
        return {};
    }
    const QJsonObject object = doc.object();
    const QJsonObject keysObject = object.value(QStringLiteral("keys")).toObject();
    if (keysObject.isEmpty()) {
        qCWarning(lcActivation) << "Plik" << path << "nie zawiera sekcji 'keys'.";
        return {};
    }
    QStringList entries;
    for (auto it = keysObject.begin(); it != keysObject.end(); ++it) {
        const QString value = it.value().toString();
        if (value.trimmed().isEmpty())
            continue;
        entries.append(QStringLiteral("%1=%2").arg(it.key(), value));
    }
    return entries;
}

QVariantList parseRegistryRecords(const QString& registryPath)
{
    QFile file(registryPath);
    if (!file.exists())
        return {};
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCWarning(lcActivation) << "Nie udało się otworzyć rejestru licencji" << registryPath << file.errorString();
        return {};
    }

    QVariantList records;
    QTextStream stream(&file);
    while (!stream.atEnd()) {
        const QString line = stream.readLine().trimmed();
        if (line.isEmpty())
            continue;
        QJsonParseError error{};
        const QJsonDocument doc = QJsonDocument::fromJson(line.toUtf8(), &error);
        if (error.error != QJsonParseError::NoError) {
            qCWarning(lcActivation) << "Nie udało się sparsować wpisu licencji:" << error.errorString();
            continue;
        }
        QVariantMap entry = doc.object().toVariantMap();
        const QVariantMap payload = entry.value(QStringLiteral("payload")).toMap();
        QVariantMap summary;
        summary.insert(QStringLiteral("licenseId"), payload.value(QStringLiteral("license_id"))); // snake_case -> camelCase
        summary.insert(QStringLiteral("issuedAt"), payload.value(QStringLiteral("issued_at")));
        summary.insert(QStringLiteral("mode"), payload.value(QStringLiteral("mode")));
        summary.insert(QStringLiteral("fingerprint"), payload.value(QStringLiteral("fingerprint")));
        summary.insert(QStringLiteral("signatureKey"), entry.value(QStringLiteral("signature")).toMap().value(QStringLiteral("key_id")));
        records.append(summary);
    }
    return records;
}

} // namespace

ActivationController::ActivationController(QObject* parent)
    : QObject(parent)
    , m_pythonExecutable(envValue("DUDZIAN_ACTIVATION_PYTHON", QStringLiteral("python3")))
    , m_keysFile(envValue("DUDZIAN_ACTIVATION_KEYS", QStringLiteral("config/oem_fingerprint_keys.json")))
    , m_rotationLog(envValue("DUDZIAN_ACTIVATION_ROTATION", QStringLiteral("var/licenses/fingerprint_rotation.json")))
    , m_registryPath(envValue("DUDZIAN_LICENSE_REGISTRY", QStringLiteral("var/licenses/registry.jsonl")))
    , m_dongleHint(envValue("DUDZIAN_ACTIVATION_DONGLE"))
{
    refresh();
}

void ActivationController::setPythonExecutable(const QString& value)
{
    if (m_pythonExecutable == value)
        return;
    m_pythonExecutable = value;
    refresh();
}

void ActivationController::setKeysFile(const QString& value)
{
    if (m_keysFile == value)
        return;
    m_keysFile = value;
    refresh();
}

void ActivationController::setRotationLog(const QString& value)
{
    if (m_rotationLog == value)
        return;
    m_rotationLog = value;
    refresh();
}

void ActivationController::setRegistryPath(const QString& value)
{
    if (m_registryPath == value)
        return;
    m_registryPath = value;
    reloadRegistry();
}

void ActivationController::setDongleHint(const QString& value)
{
    if (m_dongleHint == value)
        return;
    m_dongleHint = value;
    refresh();
}

void ActivationController::refresh()
{
    updateFingerprint();
    updateLicenses();
}

void ActivationController::reloadRegistry()
{
    updateLicenses();
}

void ActivationController::updateFingerprint()
{
    const QStringList keyArgs = loadKeyArguments();
    if (keyArgs.isEmpty()) {
        setError(tr("Brak skonfigurowanych kluczy do podpisywania fingerprintu."));
        return;
    }

    QStringList args;
    args << QStringLiteral("-m") << QStringLiteral("bot_core.security.fingerprint");
    args << QStringLiteral("--rotation-log") << m_rotationLog;
    args << QStringLiteral("--purpose") << QStringLiteral("hardware-fingerprint");
    args << QStringLiteral("--interval-days") << QStringLiteral("180");
    if (!m_dongleHint.isEmpty())
        args << QStringLiteral("--dongle") << m_dongleHint;
    for (const QString& entry : keyArgs)
        args << QStringLiteral("--key") << entry;

    QProcess process;
    process.start(m_pythonExecutable, args);
    if (!process.waitForFinished(10000)) {
        process.kill();
        setError(tr("Timeout podczas generowania fingerprintu."));
        return;
    }

    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        const QString stderrText = QString::fromUtf8(process.readAllStandardError());
        setError(tr("Nie udało się wygenerować fingerprintu: %1").arg(stderrText.trimmed()));
        return;
    }

    const QByteArray output = process.readAllStandardOutput();
    QJsonParseError error{};
    const QJsonDocument doc = QJsonDocument::fromJson(output, &error);
    if (error.error != QJsonParseError::NoError) {
        setError(tr("Nie udało się sparsować JSON fingerprintu: %1").arg(error.errorString()));
        return;
    }

    QVariantMap map = doc.object().toVariantMap();
    QVariantMap payload = map.value(QStringLiteral("payload")).toMap();
    payload = enrichFingerprintPayload(payload);
    map.insert(QStringLiteral("payload"), payload);

    m_fingerprint = map;
    Q_EMIT fingerprintChanged();
    clearError();
}

void ActivationController::updateLicenses()
{
    m_licenses = parseRegistryRecords(m_registryPath);
    Q_EMIT licensesChanged();
}

void ActivationController::setError(const QString& message)
{
    if (m_lastError == message)
        return;
    m_lastError = message;
    qCWarning(lcActivation) << message;
    Q_EMIT errorChanged();
}

void ActivationController::clearError()
{
    if (m_lastError.isEmpty())
        return;
    m_lastError.clear();
    Q_EMIT errorChanged();
}

QStringList ActivationController::loadKeyArguments() const
{
    return readKeyArguments(m_keysFile);
}

QVariantMap ActivationController::enrichFingerprintPayload(const QVariantMap& payload) const
{
    QVariantMap enriched = payload;
    const QVariantMap components = payload.value(QStringLiteral("components")).toMap();
    QVariantList componentList;
    for (auto it = components.begin(); it != components.end(); ++it) {
        QVariantMap entry = it.value().toMap();
        entry.insert(QStringLiteral("name"), it.key());
        componentList.append(entry);
    }
    enriched.insert(QStringLiteral("component_list"), componentList);
    return enriched;
}

