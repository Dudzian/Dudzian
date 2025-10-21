#include "LicenseActivationController.hpp"

#include <QByteArray>
#include <algorithm>
#include <QDate>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QSaveFile>
#include <QTimer>
#include <QtGlobal>

#include "utils/PathUtils.hpp"

Q_LOGGING_CATEGORY(lcActivation, "bot.shell.license")

using bot::shell::utils::watchableDirectories;

namespace {
QString normalizeFingerprint(const QString& value)
{
    QString trimmed = value.trimmed().toUpper();
    return trimmed;
}

QString normalizeIso(const QDateTime& dt)
{
    if (!dt.isValid())
        return {};
    return dt.toUTC().toString(Qt::ISODate);
}

QJsonDocument parseJson(const QByteArray& data, QString* errorMessage)
{
    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        if (errorMessage)
            *errorMessage = parseError.errorString();
        return {};
    }
    return doc;
}

QString extractFingerprintFromDocument(const QJsonDocument& doc)
{
    if (!doc.isObject())
        return {};
    const QJsonObject root = doc.object();
    const QString payloadB64 = root.value(QStringLiteral("payload_b64")).toString();
    if (!payloadB64.isEmpty()) {
        const QByteArray decoded = QByteArray::fromBase64(payloadB64.toUtf8(), QByteArray::Base64Option::IgnoreBase64Whitespace);
        if (!decoded.isEmpty()) {
            QString parseError;
            const QJsonDocument payloadDoc = parseJson(decoded, &parseError);
            if (!payloadDoc.isNull() && payloadDoc.isObject()) {
                const QString hwid = payloadDoc.object().value(QStringLiteral("hwid")).toString();
                if (!hwid.isEmpty())
                    return normalizeFingerprint(hwid);
            }
        }
    }
    if (root.contains(QStringLiteral("payload"))) {
        const QJsonValue payloadValue = root.value(QStringLiteral("payload"));
        if (payloadValue.isObject()) {
            const QString fp = payloadValue.toObject().value(QStringLiteral("fingerprint")).toString();
            if (!fp.isEmpty())
                return normalizeFingerprint(fp);
        }
    }
    const QString direct = root.value(QStringLiteral("fingerprint")).toString();
    if (!direct.isEmpty())
        return normalizeFingerprint(direct);
    return {};
}

QString extractFingerprintFromBytes(const QByteArray& data)
{
    QString error;
    const QJsonDocument doc = parseJson(data, &error);
    if (!doc.isNull()) {
        const QString fingerprint = extractFingerprintFromDocument(doc);
        if (!fingerprint.isEmpty())
            return fingerprint;
    }
    const QString text = QString::fromUtf8(data).trimmed();
    if (!text.isEmpty())
        return normalizeFingerprint(text);
    return {};
}

QByteArray tryDecodeBase64(const QString& text)
{
    const QByteArray raw = text.toUtf8();
    QByteArray decoded = QByteArray::fromBase64(raw, QByteArray::Base64Option::IgnoreBase64Whitespace);
    if (decoded.isEmpty())
        return {};
    return decoded;
}

} // namespace

LicenseActivationController::LicenseActivationController(QObject* parent)
    : QObject(parent)
{
    m_provisioningScanTimer.setSingleShot(true);
    m_provisioningScanTimer.setInterval(150);
    connect(&m_provisioningScanTimer, &QTimer::timeout, this, [this]() {
        attemptAutomaticProvisioning(m_pendingProvisioningError);
        m_pendingProvisioningError = false;
    });
    connect(&m_provisioningWatcher, &QFileSystemWatcher::directoryChanged, this,
            &LicenseActivationController::handleProvisioningDirectoryEvent);
    connect(&m_provisioningWatcher, &QFileSystemWatcher::fileChanged, this,
            &LicenseActivationController::handleProvisioningDirectoryEvent);

    m_licenseReloadTimer.setSingleShot(true);
    m_licenseReloadTimer.setInterval(100);
    connect(&m_licenseReloadTimer, &QTimer::timeout, this, [this]() {
        loadPersistedLicense();
    });

    m_fingerprintReloadTimer.setSingleShot(true);
    m_fingerprintReloadTimer.setInterval(75);
    connect(&m_fingerprintReloadTimer, &QTimer::timeout, this, [this]() {
        refreshExpectedFingerprint();
        if (!m_licenseActive)
            attemptAutomaticProvisioning(false);
    });

    connect(&m_licenseWatcher, &QFileSystemWatcher::fileChanged, this,
            &LicenseActivationController::handleLicensePathEvent);
    connect(&m_licenseWatcher, &QFileSystemWatcher::directoryChanged, this,
            &LicenseActivationController::handleLicensePathEvent);

    connect(&m_fingerprintWatcher, &QFileSystemWatcher::fileChanged, this,
            &LicenseActivationController::handleFingerprintPathEvent);
    connect(&m_fingerprintWatcher, &QFileSystemWatcher::directoryChanged, this,
            &LicenseActivationController::handleFingerprintPathEvent);
}

void LicenseActivationController::setConfigDirectory(const QString& path)
{
    m_configDirectory = expandPath(path);
    if (m_initialized) {
        refreshExpectedFingerprint();
        setupFingerprintWatcher();
    }
}

void LicenseActivationController::setLicenseStoragePath(const QString& path)
{
    m_licenseOutputPath = expandPath(path);
    if (m_initialized) {
        loadPersistedLicense();
        setupLicenseWatcher();
    }
}

void LicenseActivationController::setFingerprintDocumentPath(const QString& path)
{
    m_fingerprintDocumentPath = expandPath(path);
    if (m_initialized) {
        refreshExpectedFingerprint();
        setupFingerprintWatcher();
    }
}

void LicenseActivationController::setProvisioningDirectory(const QString& path)
{
    const QString expanded = expandPath(path);
    if (m_provisioningDirectory == expanded)
        return;
    m_provisioningDirectory = expanded;
    Q_EMIT provisioningDirectoryChanged();
    if (m_initialized)
        setupProvisioningWatcher();
}

void LicenseActivationController::initialize()
{
    if (m_initialized)
        return;

    if (m_licenseOutputPath.isEmpty()) {
        const QByteArray env = qgetenv("BOT_CORE_UI_ACTIVE_LICENSE_PATH");
        if (!env.isEmpty())
            m_licenseOutputPath = expandPath(QString::fromUtf8(env));
    }
    if (m_licenseOutputPath.isEmpty())
        m_licenseOutputPath = expandPath(QStringLiteral("var/licenses/active/license.json"));

    if (m_fingerprintDocumentPath.isEmpty()) {
        const QByteArray env = qgetenv("BOT_CORE_UI_EXPECTED_FINGERPRINT" );
        if (!env.isEmpty())
            m_fingerprintDocumentPath = expandPath(QString::fromUtf8(env));
    }
    if (m_fingerprintDocumentPath.isEmpty()) {
        if (!m_configDirectory.isEmpty())
            m_fingerprintDocumentPath = QDir(m_configDirectory).filePath(QStringLiteral("fingerprint.expected.json"));
        else
            m_fingerprintDocumentPath = expandPath(QStringLiteral("config/fingerprint.expected.json"));
    }

    if (m_provisioningDirectory.isEmpty()) {
        const QByteArray env = qgetenv("BOT_CORE_UI_LICENSE_INBOX");
        if (!env.isEmpty())
            m_provisioningDirectory = expandPath(QString::fromUtf8(env));
    }
    if (m_provisioningDirectory.isEmpty()) {
        if (!m_configDirectory.isEmpty())
            m_provisioningDirectory = QDir(m_configDirectory).filePath(QStringLiteral("licenses/inbox"));
        else
            m_provisioningDirectory = expandPath(QStringLiteral("var/licenses/inbox"));
    }

    refreshExpectedFingerprint();
    loadPersistedLicense();

    m_initialized = true;
    Q_EMIT provisioningDirectoryChanged();

    setupProvisioningWatcher();
    setupFingerprintWatcher();
    setupLicenseWatcher();
    attemptAutomaticProvisioning(false);
}

bool LicenseActivationController::ensureInitialized()
{
    if (!m_initialized)
        initialize();
    return m_initialized;
}

QString LicenseActivationController::licenseStoragePath() const
{
    return resolveLicenseOutputPath();
}

bool LicenseActivationController::saveExpectedFingerprint(const QString& fingerprint)
{
    if (!ensureInitialized())
        return false;

    const QString normalized = normalizeFingerprint(fingerprint);
    if (normalized.isEmpty()) {
        setStatusMessage(tr("Fingerprint nie może być pusty"), true);
        return false;
    }

    QString errorMessage;
    if (!persistExpectedFingerprint(normalized, &errorMessage)) {
        setStatusMessage(errorMessage, true);
        return false;
    }

    if (m_expectedFingerprint != normalized) {
        m_expectedFingerprint = normalized;
        Q_EMIT expectedFingerprintChanged();
    }
    setStatusMessage(tr("Zapisano oczekiwany fingerprint: %1").arg(normalized), false);
    return true;
}

void LicenseActivationController::overrideExpectedFingerprint(const QString& fingerprint)
{
    const QString normalized = normalizeFingerprint(fingerprint);
    if (normalized.isEmpty())
        return;
    if (m_expectedFingerprint == normalized)
        return;
    m_expectedFingerprint = normalized;
    Q_EMIT expectedFingerprintChanged();
}

QString LicenseActivationController::resolveLicenseOutputPath() const
{
    if (!m_licenseOutputPath.isEmpty())
        return m_licenseOutputPath;
    return expandPath(QStringLiteral("var/licenses/active/license.json"));
}

QString LicenseActivationController::resolveFingerprintDocumentPath() const
{
    if (!m_fingerprintDocumentPath.isEmpty())
        return m_fingerprintDocumentPath;
    if (!m_configDirectory.isEmpty())
        return QDir(m_configDirectory).filePath(QStringLiteral("fingerprint.expected.json"));
    return expandPath(QStringLiteral("config/fingerprint.expected.json"));
}

QString LicenseActivationController::resolveProvisioningDirectory() const
{
    if (!m_provisioningDirectory.isEmpty())
        return m_provisioningDirectory;
    if (!m_configDirectory.isEmpty())
        return QDir(m_configDirectory).filePath(QStringLiteral("licenses/inbox"));
    return expandPath(QStringLiteral("var/licenses/inbox"));
}

bool LicenseActivationController::loadLicenseUrl(const QUrl& url)
{
    if (!ensureInitialized())
        return false;
    if (!url.isValid()) {
        setStatusMessage(tr("Nieprawidłowy adres pliku licencji"), true);
        return false;
    }
    if (url.isLocalFile())
        return loadLicenseFile(url.toLocalFile());
    return loadLicenseFile(url.toString());
}

bool LicenseActivationController::loadLicenseFile(const QString& path)
{
    if (!ensureInitialized())
        return false;
    const QString expanded = expandPath(path);
    QFile file(expanded);
    if (!file.exists()) {
        setStatusMessage(tr("Plik licencji nie istnieje: %1").arg(expanded), true);
        return false;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        setStatusMessage(tr("Nie można odczytać pliku licencji %1: %2").arg(expanded, file.errorString()), true);
        return false;
    }
    const QByteArray data = file.readAll();
    file.close();

    QString error;
    QJsonDocument doc = parseJson(data, &error);
    if (doc.isNull()) {
        setStatusMessage(tr("Niepoprawny dokument licencji (%1)").arg(error), true);
        return false;
    }
    return activateFromDocument(doc, true, expanded);
}

bool LicenseActivationController::applyLicenseText(const QString& text)
{
    if (!ensureInitialized())
        return false;
    const QString trimmed = text.trimmed();
    if (trimmed.isEmpty()) {
        setStatusMessage(tr("Wklejony payload jest pusty"), true);
        return false;
    }

    QString error;
    QJsonDocument doc = parseJson(trimmed.toUtf8(), &error);
    if (doc.isNull()) {
        const QByteArray decoded = tryDecodeBase64(trimmed);
        if (decoded.isEmpty()) {
            setStatusMessage(tr("Nie udało się zdekodować payloadu (błąd JSON: %1)").arg(error), true);
            return false;
        }
        error.clear();
        doc = parseJson(decoded, &error);
        if (doc.isNull()) {
            setStatusMessage(tr("Dekodowanie base64 powiodło się, ale JSON nadal jest niepoprawny: %1").arg(error), true);
            return false;
        }
    }

    return activateFromDocument(doc, true, tr("payload"));
}

void LicenseActivationController::refreshExpectedFingerprint()
{
    const QString path = resolveFingerprintDocumentPath();
    QFile file(path);
    if (!file.exists()) {
        if (!m_expectedFingerprint.isEmpty()) {
            m_expectedFingerprint.clear();
            Q_EMIT expectedFingerprintChanged();
        }
        return;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (!m_expectedFingerprint.isEmpty()) {
            m_expectedFingerprint.clear();
            Q_EMIT expectedFingerprintChanged();
        }
        return;
    }
    const QByteArray data = file.readAll();
    file.close();
    const QString fingerprint = extractFingerprintFromBytes(data);
    if (fingerprint != m_expectedFingerprint) {
        m_expectedFingerprint = fingerprint;
        Q_EMIT expectedFingerprintChanged();
    }
}

void LicenseActivationController::loadPersistedLicense()
{
    const QString path = resolveLicenseOutputPath();
    QFile file(path);
    if (!file.exists()) {
        clearLicenseState();
        return;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        setStatusMessage(tr("Nie udało się wczytać zapisanej licencji (%1)").arg(file.errorString()), true);
        clearLicenseState();
        return;
    }
    const QByteArray data = file.readAll();
    file.close();

    QString error;
    const QJsonDocument doc = parseJson(data, &error);
    if (doc.isNull()) {
        setStatusMessage(tr("Niepoprawna zapisana licencja: %1").arg(error), true);
        clearLicenseState();
        return;
    }
    activateFromDocument(doc, false, path);
}

bool LicenseActivationController::autoProvision(const QVariantMap& fingerprintDocument)
{
    if (!ensureInitialized())
        return false;

    const QString docFingerprint = fingerprintFromVariant(fingerprintDocument);
    if (!docFingerprint.isEmpty()) {
        const QString normalized = normalizeFingerprint(docFingerprint);
        if (normalized.isEmpty()) {
            setStatusMessage(tr("Fingerprint z dokumentu provisioning jest pusty"), true);
            return false;
        }
        if (!expectedFingerprintAvailable() || m_expectedFingerprint != normalized) {
            if (!saveExpectedFingerprint(normalized))
                return false;
        }
    }

    if (m_licenseActive) {
        setStatusMessage(tr("Licencja jest już aktywna"), false);
        return true;
    }

    const QString provisioningDir = expandPath(resolveProvisioningDirectory());
    QString expected = m_expectedFingerprint;
    if (expected.isEmpty())
        expected = docFingerprint;
    expected = normalizeFingerprint(expected);
    if (expected.isEmpty()) {
        setStatusMessage(tr("Automatyczna aktywacja wymaga fingerprintu urządzenia"), true);
        return false;
    }

    return runProvisioningScan(expected, true);
}

bool LicenseActivationController::activateFromDocument(const QJsonDocument& document, bool persist, const QString& sourceDescription)
{
    LicenseInfo info;
    QString error;
    if (!parseLicenseDocument(document, info, error)) {
        setStatusMessage(error, true);
        return false;
    }

    if (persist && !persistLicense(document))
        return false;

    const bool wasActive = m_licenseActive;
    m_licenseActive = true;
    m_licenseFingerprint = info.fingerprint;
    m_licenseEdition = info.edition;
    m_licenseLicenseId = info.licenseId;
    m_licenseIssuedAt = info.issuedAtIso;
    m_licenseMaintenanceUntil = info.maintenanceUntilIso;
    m_licenseMaintenanceActive = info.maintenanceActive;
    m_licenseHolderName = info.holderName;
    m_licenseHolderEmail = info.holderEmail;
    m_licenseSeats = info.seats;
    m_licenseTrialActive = info.trialActive;
    m_licenseTrialExpiresAt = info.trialExpiresIso;
    m_licenseModules = info.modules;
    m_licenseEnvironments = info.environments;
    m_licenseRuntime = info.runtime;
    m_lastDocument = info.document;

    if (!wasActive)
        Q_EMIT licenseActiveChanged();
    Q_EMIT licenseDataChanged();

    QStringList summaryParts;
    summaryParts.append(tr("edycja %1").arg(info.edition.isEmpty() ? tr("nieznana") : info.edition));
    if (!info.licenseId.isEmpty())
        summaryParts.append(tr("ID %1").arg(info.licenseId));
    if (!info.maintenanceUntilIso.isEmpty())
        summaryParts.append(tr("utrzymanie do %1").arg(info.maintenanceUntilIso));
    if (!info.fingerprint.isEmpty())
        summaryParts.append(tr("HWID %1").arg(info.fingerprint));

    QString summary = tr("Licencja aktywna (%1)").arg(summaryParts.join(QStringLiteral(", ")));
    if (!sourceDescription.isEmpty())
        summary += tr(" • źródło: %1").arg(sourceDescription);
    setStatusMessage(summary, false);
    if (persist)
        Q_EMIT licensePersisted(resolveLicenseOutputPath());
    return true;
}

bool LicenseActivationController::parseLicenseDocument(const QJsonDocument& document, LicenseInfo& info, QString& error) const
{
    if (!document.isObject()) {
        error = tr("Licencja musi być dokumentem JSON typu object");
        return false;
    }
    const QJsonObject root = document.object();
    const QString payloadB64 = root.value(QStringLiteral("payload_b64")).toString();
    if (!payloadB64.isEmpty()) {
        const QString signatureB64 = root.value(QStringLiteral("signature_b64")).toString();
        if (signatureB64.trimmed().isEmpty()) {
            error = tr("Licencja musi zawierać pole 'signature_b64'");
            return false;
        }
        const QByteArray payloadBytes = QByteArray::fromBase64(payloadB64.toUtf8(), QByteArray::Base64Option::IgnoreBase64Whitespace);
        if (payloadBytes.isEmpty()) {
            error = tr("Nie można zdekodować sekcji payload_b64 (base64)");
            return false;
        }
        QString parseErrorMessage;
        const QJsonDocument payloadDoc = parseJson(payloadBytes, &parseErrorMessage);
        if (payloadDoc.isNull() || !payloadDoc.isObject()) {
            error = tr("Payload licencji nie zawiera obiektu JSON: %1").arg(parseErrorMessage);
            return false;
        }
        const QJsonObject payload = payloadDoc.object();
        const QString edition = payload.value(QStringLiteral("edition")).toString().trimmed();
        if (edition.isEmpty()) {
            error = tr("Licencja nie zawiera pola 'edition'");
            return false;
        }
        const QString hwid = normalizeFingerprint(payload.value(QStringLiteral("hwid")).toString());
        if (expectedFingerprintAvailable()) {
            if (hwid.isEmpty()) {
                error = tr("Licencja nie zawiera fingerprintu HWID, oczekiwano %1").arg(m_expectedFingerprint);
                return false;
            }
            if (hwid != m_expectedFingerprint) {
                error = tr("Fingerprint licencji (%1) nie zgadza się z oczekiwanym (%2)")
                            .arg(hwid, m_expectedFingerprint);
                return false;
            }
        }

        info.fingerprint = hwid;
        info.licenseId = payload.value(QStringLiteral("license_id")).toString();
        info.edition = edition;
        info.issuedAtIso = payload.value(QStringLiteral("issued_at")).toString();
        info.maintenanceUntilIso = payload.value(QStringLiteral("maintenance_until")).toString();
        const QDate maintenanceDate = QDate::fromString(info.maintenanceUntilIso, Qt::ISODate);
        if (maintenanceDate.isValid())
            info.maintenanceActive = maintenanceDate >= QDate::currentDate();
        else
            info.maintenanceActive = true;

        const QJsonObject holderObj = payload.value(QStringLiteral("holder")).toObject();
        info.holderName = holderObj.value(QStringLiteral("name")).toString();
        info.holderEmail = holderObj.value(QStringLiteral("email")).toString();
        bool okSeats = false;
        info.seats = payload.value(QStringLiteral("seats")).toInt(&okSeats);
        if (!okSeats)
            info.seats = 0;

        const QJsonObject trialObj = payload.value(QStringLiteral("trial")).toObject();
        info.trialActive = trialObj.value(QStringLiteral("enabled")).toBool();
        info.trialExpiresIso = trialObj.value(QStringLiteral("expires_at")).toString();

        const auto collectEnabled = [](const QJsonValue& value) {
            QStringList enabled;
            if (value.isObject()) {
                const QJsonObject obj = value.toObject();
                for (auto it = obj.constBegin(); it != obj.constEnd(); ++it) {
                    if (it.value().toBool())
                        enabled.append(it.key());
                }
            }
            std::sort(enabled.begin(), enabled.end());
            return enabled;
        };

        info.modules = collectEnabled(payload.value(QStringLiteral("modules")));
        info.runtime = collectEnabled(payload.value(QStringLiteral("runtime")));

        const QJsonValue envValue = payload.value(QStringLiteral("environments"));
        if (envValue.isArray()) {
            const QJsonArray array = envValue.toArray();
            QStringList environments;
            environments.reserve(array.size());
            for (const QJsonValue& env : array) {
                if (env.isString()) {
                    const QString trimmed = env.toString().trimmed();
                    if (!trimmed.isEmpty())
                        environments.append(trimmed);
                }
            }
            std::sort(environments.begin(), environments.end());
            info.environments = environments;
        }

        info.document = document;
        return true;
    }

    const QJsonValue payloadValue = root.value(QStringLiteral("payload"));
    const QJsonValue signatureValue = root.value(QStringLiteral("signature"));
    if (!payloadValue.isObject() || !signatureValue.isObject()) {
        error = tr("Licencja musi zawierać sekcje 'payload' oraz 'signature'");
        return false;
    }
    const QJsonObject payload = payloadValue.toObject();
    const QJsonObject signature = signatureValue.toObject();

    const QString schema = payload.value(QStringLiteral("schema")).toString();
    if (schema != QStringLiteral("core.oem.license")) {
        error = tr("Nieobsługiwany typ dokumentu licencyjnego: %1").arg(schema);
        return false;
    }

    const QString fingerprint = normalizeFingerprint(payload.value(QStringLiteral("fingerprint")).toString());
    if (fingerprint.isEmpty()) {
        error = tr("Brak fingerprintu w payloadzie licencji");
        return false;
    }
    if (expectedFingerprintAvailable() && fingerprint != m_expectedFingerprint) {
        error = tr("Fingerprint licencji (%1) nie zgadza się z oczekiwanym (%2)")
                    .arg(fingerprint, m_expectedFingerprint);
        return false;
    }

    const QString profile = payload.value(QStringLiteral("profile")).toString();
    if (profile.isEmpty()) {
        error = tr("Brak profilu pracy w licencji");
        return false;
    }

    const QString issuedAtRaw = payload.value(QStringLiteral("issued_at")).toString();
    const QString expiresAtRaw = payload.value(QStringLiteral("expires_at")).toString();
    if (expiresAtRaw.isEmpty()) {
        error = tr("Licencja nie zawiera pola 'expires_at'");
        return false;
    }

    QDateTime expiresAt = QDateTime::fromString(expiresAtRaw, Qt::ISODate);
    if (!expiresAt.isValid())
        expiresAt = QDateTime::fromString(expiresAtRaw, Qt::ISODateWithMs);
    if (!expiresAt.isValid()) {
        error = tr("Nieprawidłowy format daty 'expires_at': %1").arg(expiresAtRaw);
        return false;
    }
    expiresAt = expiresAt.toUTC();
    if (expiresAt < QDateTime::currentDateTimeUtc()) {
        error = tr("Licencja wygasła (%1)").arg(expiresAt.toString(Qt::ISODate));
        return false;
    }

    QDateTime issuedAt;
    if (!issuedAtRaw.isEmpty()) {
        issuedAt = QDateTime::fromString(issuedAtRaw, Qt::ISODate);
        if (!issuedAt.isValid())
            issuedAt = QDateTime::fromString(issuedAtRaw, Qt::ISODateWithMs);
    }

    const QJsonValue featuresValue = payload.value(QStringLiteral("features"));
    QStringList features;
    if (featuresValue.isArray()) {
        const QJsonArray array = featuresValue.toArray();
        for (const QJsonValue& featureValue : array) {
            if (!featureValue.isString()) {
                error = tr("Pole 'features' musi zawierać listę stringów");
                return false;
            }
            const QString feature = featureValue.toString().trimmed();
            if (!feature.isEmpty())
                features.append(feature);
        }
        std::sort(features.begin(), features.end());
    } else if (!featuresValue.isUndefined() && !featuresValue.isNull()) {
        error = tr("Pole 'features' w licencji musi być tablicą");
        return false;
    }

    const QString algorithm = signature.value(QStringLiteral("algorithm")).toString();
    if (algorithm.trimmed().isEmpty()) {
        error = tr("Brak algorytmu podpisu w licencji");
        return false;
    }
    if (!algorithm.startsWith(QStringLiteral("HMAC"), Qt::CaseInsensitive)) {
        error = tr("Nieobsługiwany algorytm podpisu licencji: %1").arg(algorithm);
        return false;
    }
    const QString signatureValue = signature.value(QStringLiteral("value")).toString();
    if (signatureValue.trimmed().isEmpty()) {
        error = tr("Podpis licencji jest pusty");
        return false;
    }

    info.fingerprint = fingerprint;
    info.edition = profile;
    info.licenseId = payload.value(QStringLiteral("license_id")).toString();
    info.issuedAtIso = issuedAt.isValid() ? normalizeIso(issuedAt) : issuedAtRaw;
    info.maintenanceUntilIso = normalizeIso(expiresAt);
    info.maintenanceActive = true;
    info.modules = features;
    info.environments = {};
    info.runtime = {};
    info.document = document;

    return true;
}

bool LicenseActivationController::persistLicense(const QJsonDocument& document)
{
    const QString path = resolveLicenseOutputPath();
    if (path.isEmpty()) {
        setStatusMessage(tr("Brak skonfigurowanej ścieżki do zapisu licencji"), true);
        return false;
    }
    QFileInfo info(path);
    QDir dir = info.dir();
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) {
        setStatusMessage(tr("Nie udało się utworzyć katalogu dla licencji: %1").arg(dir.path()), true);
        return false;
    }

    QSaveFile file(path);
    file.setDirectWriteFallback(true);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        setStatusMessage(tr("Nie można zapisać licencji do %1: %2").arg(path, file.errorString()), true);
        return false;
    }
    const QByteArray payload = document.toJson(QJsonDocument::Indented);
    if (file.write(payload) != payload.size()) {
        setStatusMessage(tr("Błąd zapisu licencji do %1: %2").arg(path, file.errorString()), true);
        return false;
    }
    if (!file.commit()) {
        setStatusMessage(tr("Nie udało się zatwierdzić zapisu licencji: %1").arg(file.errorString()), true);
        return false;
    }
    return true;
}

void LicenseActivationController::setStatusMessage(const QString& message, bool isError)
{
    if (m_statusMessage == message && m_statusIsError == isError)
        return;
    m_statusMessage = message;
    m_statusIsError = isError;
    Q_EMIT statusMessageChanged();
}

QString LicenseActivationController::expandPath(const QString& path)
{
    return bot::shell::utils::expandPath(path);
}

bool LicenseActivationController::persistExpectedFingerprint(const QString& fingerprint, QString* errorMessage)
{
    const QString path = resolveFingerprintDocumentPath();
    if (path.isEmpty()) {
        if (errorMessage)
            *errorMessage = tr("Brak ścieżki docelowej dla fingerprintu");
        return false;
    }

    QFileInfo info(path);
    QDir dir = info.dir();
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) {
        if (errorMessage)
            *errorMessage = tr("Nie udało się utworzyć katalogu %1").arg(dir.path());
        return false;
    }

    QJsonObject root;
    root.insert(QStringLiteral("fingerprint"), fingerprint);
    root.insert(QStringLiteral("updated_at"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
    QJsonDocument document(root);

    QSaveFile file(path);
    file.setDirectWriteFallback(true);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (errorMessage)
            *errorMessage = tr("Nie można zapisać fingerprintu do %1: %2").arg(path, file.errorString());
        return false;
    }

    const QByteArray payload = document.toJson(QJsonDocument::Indented);
    if (file.write(payload) != payload.size()) {
        if (errorMessage)
            *errorMessage = tr("Błąd zapisu fingerprintu: %1").arg(file.errorString());
        return false;
    }
    if (!file.commit()) {
        if (errorMessage)
            *errorMessage = tr("Nie udało się zatwierdzić pliku fingerprintu: %1").arg(file.errorString());
        return false;
    }
    return true;
}

QString LicenseActivationController::fingerprintFromVariant(const QVariantMap& fingerprintDocument)
{
    if (fingerprintDocument.isEmpty())
        return {};
    const QVariantMap payload = fingerprintDocument.value(QStringLiteral("payload")).toMap();
    QString fingerprint = payload.value(QStringLiteral("fingerprint")).toString();
    if (fingerprint.isEmpty())
        fingerprint = fingerprintDocument.value(QStringLiteral("fingerprint")).toString();
    return normalizeFingerprint(fingerprint);
}

void LicenseActivationController::setupProvisioningWatcher()
{
    const QStringList watchedDirs = m_provisioningWatcher.directories();
    for (const QString& dir : watchedDirs)
        m_provisioningWatcher.removePath(dir);
    const QStringList watchedFiles = m_provisioningWatcher.files();
    for (const QString& file : watchedFiles)
        m_provisioningWatcher.removePath(file);

    const QString directory = expandPath(resolveProvisioningDirectory());
    if (directory.isEmpty())
        return;

    QDir dir(directory);
    if (!dir.exists())
        dir.mkpath(QStringLiteral("."));

    m_provisioningWatcher.addPath(directory);
    scheduleProvisioningScan(0, false);
}

void LicenseActivationController::handleProvisioningDirectoryEvent(const QString& path)
{
    Q_UNUSED(path);
    scheduleProvisioningScan(200, false);
}

void LicenseActivationController::scheduleProvisioningScan(int delayMs, bool reportNotFound)
{
    if (delayMs < 0)
        delayMs = 0;
    m_pendingProvisioningError = reportNotFound;
    if (m_provisioningScanTimer.isActive())
        m_provisioningScanTimer.stop();
    m_provisioningScanTimer.start(delayMs);
}

void LicenseActivationController::attemptAutomaticProvisioning(bool reportNotFound)
{
    if (!m_initialized)
        return;
    if (m_licenseActive)
        return;

    QString expected = m_expectedFingerprint;
    if (expected.isEmpty()) {
        refreshExpectedFingerprint();
        expected = m_expectedFingerprint;
    }

    if (expected.isEmpty())
        return;

    runProvisioningScan(expected, reportNotFound);
}

bool LicenseActivationController::runProvisioningScan(const QString& expectedFingerprint, bool reportNotFound)
{
    const QString provisioningDir = expandPath(resolveProvisioningDirectory());
    if (provisioningDir.isEmpty())
        return false;

    const QString normalized = normalizeFingerprint(expectedFingerprint);
    if (normalized.isEmpty())
        return false;

    QDir dir(provisioningDir);
    if (!dir.exists())
        dir.mkpath(QStringLiteral("."));

    const bool previousState = m_provisioningInProgress;
    if (!previousState) {
        m_provisioningInProgress = true;
        Q_EMIT provisioningInProgressChanged();
    }

    const bool activated = provisionFromDirectory(provisioningDir, normalized);

    if (!previousState) {
        m_provisioningInProgress = false;
        Q_EMIT provisioningInProgressChanged();
    }

    if (activated)
        return true;

    if (reportNotFound) {
        setStatusMessage(tr("Nie znaleziono licencji dopasowanej do fingerprintu %1 w katalogu %2")
                             .arg(normalized, provisioningDir),
                         true);
    } else {
        qCDebug(lcActivation) << "Provisioning scan completed without match" << provisioningDir;
    }

    return false;
}

bool LicenseActivationController::provisionFromDirectory(const QString& directory, const QString& expectedFingerprint)
{
    QDir dir(directory);
    if (!dir.exists())
        return false;

    const QStringList filters = {QStringLiteral("*.json"), QStringLiteral("*.jsonl"),
                                 QStringLiteral("*.lic"), QStringLiteral("*.txt"), QStringLiteral("*.payload")};
    const QFileInfoList files = dir.entryInfoList(filters, QDir::Files | QDir::Readable);
    for (const QFileInfo& info : files) {
        if (tryProvisionFile(info.absoluteFilePath(), expectedFingerprint))
            return true;
    }
    return false;
}

bool LicenseActivationController::tryProvisionFile(const QString& path, const QString& expectedFingerprint)
{
    QFile file(path);
    if (!file.exists())
        return false;
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;
    const QByteArray data = file.readAll();
    file.close();

    auto finalizeFile = [path]() {
        const QString archived = path + QStringLiteral(".applied");
        if (QFile::exists(archived))
            QFile::remove(archived);
        if (!QFile::rename(path, archived))
            qCWarning(lcActivation) << "Nie udało się oznaczyć przetworzonej licencji" << path;
    };

    QString parseError;
    QJsonDocument doc = parseJson(data, &parseError);
    if (!doc.isNull() && activateIfMatching(doc, true, path, expectedFingerprint)) {
        finalizeFile();
        return true;
    }

    const QList<QByteArray> lines = data.split('\n');
    int lineNumber = 0;
    for (const QByteArray& rawLine : lines) {
        ++lineNumber;
        const QByteArray trimmed = rawLine.trimmed();
        if (trimmed.isEmpty())
            continue;
        QString error;
        QJsonDocument lineDoc = parseJson(trimmed, &error);
        if (!lineDoc.isNull() && activateIfMatching(lineDoc, true,
                                                    QStringLiteral("%1:%2").arg(path).arg(lineNumber), expectedFingerprint)) {
            finalizeFile();
            return true;
        }
    }

    const QByteArray decoded = tryDecodeBase64(QString::fromUtf8(data));
    if (!decoded.isEmpty()) {
        QString error;
        const QJsonDocument decodedDoc = parseJson(decoded, &error);
        if (!decodedDoc.isNull() && activateIfMatching(decodedDoc, true, path, expectedFingerprint)) {
            finalizeFile();
            return true;
        }
    }

    return false;
}

bool LicenseActivationController::activateIfMatching(const QJsonDocument& document, bool persist,
                                                     const QString& sourceDescription,
                                                     const QString& expectedFingerprint)
{
    LicenseInfo info;
    QString error;
    if (!parseLicenseDocument(document, info, error))
        return false;
    const QString normalizedExpected = normalizeFingerprint(expectedFingerprint);
    if (!normalizedExpected.isEmpty() && normalizeFingerprint(info.fingerprint) != normalizedExpected)
        return false;
    return activateFromDocument(document, persist, sourceDescription);
}

void LicenseActivationController::setupLicenseWatcher()
{
    const QStringList currentFiles = m_licenseWatcher.files();
    if (!currentFiles.isEmpty())
        m_licenseWatcher.removePaths(currentFiles);
    const QStringList currentDirs = m_licenseWatcher.directories();
    if (!currentDirs.isEmpty())
        m_licenseWatcher.removePaths(currentDirs);

    const QString path = resolveLicenseOutputPath();
    if (path.isEmpty())
        return;

    QFileInfo info(path);
    const QString absolutePath = info.absoluteFilePath();
    if (info.exists() && info.isFile())
        m_licenseWatcher.addPath(absolutePath);

    const QString directory = info.absolutePath();
    const QStringList directories = watchableDirectories(directory);
    for (const QString& dir : directories) {
        if (!dir.isEmpty())
            m_licenseWatcher.addPath(dir);
    }

    if (!directory.isEmpty()) {
        QDir dir(directory);
        if (!dir.exists())
            dir.mkpath(QStringLiteral("."));
    }
}

void LicenseActivationController::setupFingerprintWatcher()
{
    const QStringList currentFiles = m_fingerprintWatcher.files();
    if (!currentFiles.isEmpty())
        m_fingerprintWatcher.removePaths(currentFiles);
    const QStringList currentDirs = m_fingerprintWatcher.directories();
    if (!currentDirs.isEmpty())
        m_fingerprintWatcher.removePaths(currentDirs);

    const QString path = resolveFingerprintDocumentPath();
    if (path.isEmpty())
        return;

    QFileInfo info(path);
    const QString absolutePath = info.absoluteFilePath();
    if (info.exists() && info.isFile())
        m_fingerprintWatcher.addPath(absolutePath);

    const QString directory = info.absolutePath();
    const QStringList directories = watchableDirectories(directory);
    for (const QString& dir : directories) {
        if (!dir.isEmpty())
            m_fingerprintWatcher.addPath(dir);
    }
}

void LicenseActivationController::handleLicensePathEvent(const QString& path)
{
    Q_UNUSED(path);
    setupLicenseWatcher();
    scheduleLicenseReload(120);
}

void LicenseActivationController::handleFingerprintPathEvent(const QString& path)
{
    Q_UNUSED(path);
    setupFingerprintWatcher();
    scheduleFingerprintReload(100);
}

void LicenseActivationController::scheduleLicenseReload(int delayMs)
{
    if (delayMs < 0)
        delayMs = 0;
    m_licenseReloadTimer.start(delayMs);
}

void LicenseActivationController::scheduleFingerprintReload(int delayMs)
{
    if (delayMs < 0)
        delayMs = 0;
    m_fingerprintReloadTimer.start(delayMs);
}

void LicenseActivationController::clearLicenseState()
{
    const bool wasActive = m_licenseActive;
    const bool hadData = wasActive || !m_licenseFingerprint.isEmpty() || !m_licenseEdition.isEmpty() ||
                         !m_licenseLicenseId.isEmpty() || !m_licenseIssuedAt.isEmpty() ||
                         !m_licenseMaintenanceUntil.isEmpty() || m_licenseMaintenanceActive ||
                         !m_licenseHolderName.isEmpty() || !m_licenseHolderEmail.isEmpty() || m_licenseSeats != 0 ||
                         m_licenseTrialActive || !m_licenseTrialExpiresAt.isEmpty() || !m_licenseModules.isEmpty() ||
                         !m_licenseEnvironments.isEmpty() || !m_licenseRuntime.isEmpty() || !m_lastDocument.isNull();

    if (!hadData)
        return;

    m_licenseActive = false;
    m_licenseFingerprint.clear();
    m_licenseEdition.clear();
    m_licenseLicenseId.clear();
    m_licenseIssuedAt.clear();
    m_licenseMaintenanceUntil.clear();
    m_licenseMaintenanceActive = false;
    m_licenseHolderName.clear();
    m_licenseHolderEmail.clear();
    m_licenseSeats = 0;
    m_licenseTrialActive = false;
    m_licenseTrialExpiresAt.clear();
    m_licenseModules.clear();
    m_licenseEnvironments.clear();
    m_licenseRuntime.clear();
    m_lastDocument = QJsonDocument();

    if (wasActive)
        Q_EMIT licenseActiveChanged();
    Q_EMIT licenseDataChanged();
    if (!m_statusIsError)
        setStatusMessage(tr("Brak aktywnej licencji"), false);
}
