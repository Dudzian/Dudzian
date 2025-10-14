#include "LicenseActivationController.hpp"

#include <QByteArray>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QSaveFile>
#include <QtGlobal>

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
}

void LicenseActivationController::setConfigDirectory(const QString& path)
{
    m_configDirectory = expandPath(path);
    if (m_initialized)
        refreshExpectedFingerprint();
}

void LicenseActivationController::setLicenseStoragePath(const QString& path)
{
    m_licenseOutputPath = expandPath(path);
    if (m_initialized)
        loadPersistedLicense();
}

void LicenseActivationController::setFingerprintDocumentPath(const QString& path)
{
    m_fingerprintDocumentPath = expandPath(path);
    if (m_initialized)
        refreshExpectedFingerprint();
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

    refreshExpectedFingerprint();
    loadPersistedLicense();

    m_initialized = true;
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
        return;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        setStatusMessage(tr("Nie udało się wczytać zapisanej licencji (%1)").arg(file.errorString()), true);
        return;
    }
    const QByteArray data = file.readAll();
    file.close();

    QString error;
    const QJsonDocument doc = parseJson(data, &error);
    if (doc.isNull()) {
        setStatusMessage(tr("Niepoprawna zapisana licencja: %1").arg(error), true);
        return;
    }
    activateFromDocument(doc, false, path);
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
    m_licenseProfile = info.profile;
    m_licenseIssuer = info.issuer;
    m_licenseBundleVersion = info.bundleVersion;
    m_licenseIssuedAt = info.issuedAtIso;
    m_licenseExpiresAt = info.expiresAtIso;
    m_licenseFeatures = info.features;
    m_lastDocument = info.document;

    if (!wasActive)
        Q_EMIT licenseActiveChanged();
    Q_EMIT licenseDataChanged();

    QString summary = tr("Licencja aktywna: profil %1, ważna do %2 (%3)")
                        .arg(info.profile, info.expiresAtIso, info.fingerprint);
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

    const QString issuer = payload.value(QStringLiteral("issuer")).toString();
    if (issuer.isEmpty()) {
        error = tr("Brak identyfikatora wystawcy w licencji");
        return false;
    }

    const QString bundleVersion = payload.value(QStringLiteral("bundle_version")).toString();
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
    info.issuer = issuer;
    info.profile = profile;
    info.bundleVersion = bundleVersion;
    info.issuedAtIso = issuedAt.isValid() ? normalizeIso(issuedAt) : issuedAtRaw;
    info.expiresAtIso = normalizeIso(expiresAt);
    info.features = features;
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
    if (path.trimmed().isEmpty())
        return {};
    QString expanded = path.trimmed();
    if (expanded == QStringLiteral("~"))
        expanded = QDir::homePath();
    else if (expanded.startsWith(QStringLiteral("~/")))
        expanded = QDir::homePath() + expanded.mid(1);
    QFileInfo info(expanded);
    if (!info.isAbsolute())
        expanded = QDir::current().absoluteFilePath(expanded);
    return QDir::cleanPath(expanded);
}
