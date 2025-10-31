#include "SecurityAdminController.hpp"

#include <QByteArray>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLoggingCategory>
#include <QProcess>
#include <QtGlobal>
#include <utility>

#include "utils/PathUtils.hpp"

Q_LOGGING_CATEGORY(lcSecurityAdmin, "bot.shell.security.admin")

namespace {

QStringList collectIssueMessages(const QVariantMap& map,
                                 const QString& detailKey,
                                 const QString& fallbackKey)
{
    QStringList messages;
    const QVariant detailVariant = map.value(detailKey);
    if (detailVariant.canConvert<QVariantList>()) {
        const QVariantList detailList = detailVariant.toList();
        for (const QVariant& entry : detailList) {
            const QVariantMap issue = entry.toMap();
            if (issue.isEmpty())
                continue;
            QString text = issue.value(QStringLiteral("message")).toString();
            const QString hint = issue.value(QStringLiteral("hint")).toString();
            const QString code = issue.value(QStringLiteral("code")).toString();
            if (!hint.isEmpty())
                text += QStringLiteral(" — %1").arg(hint);
            if (!code.isEmpty())
                text += QStringLiteral(" [%1]").arg(code);
            if (!text.isEmpty())
                messages.append(text);
        }
    }
    if (messages.isEmpty()) {
        const QVariant fallbackVariant = map.value(fallbackKey);
        if (fallbackVariant.canConvert<QStringList>()) {
            messages = fallbackVariant.toStringList();
        } else if (fallbackVariant.canConvert<QVariantList>()) {
            const QVariantList fallbackList = fallbackVariant.toList();
            for (const QVariant& entry : fallbackList) {
                if (entry.canConvert<QString>()) {
                    const QString text = entry.toString();
                    if (!text.isEmpty())
                        messages.append(text);
                } else if (entry.canConvert<QVariantMap>()) {
                    const QVariantMap issue = entry.toMap();
                    const QString text = issue.value(QStringLiteral("message")).toString();
                    if (!text.isEmpty())
                        messages.append(text);
                }
            }
        } else if (fallbackVariant.canConvert<QString>()) {
            const QString text = fallbackVariant.toString();
            if (!text.isEmpty())
                messages.append(text);
        }
    }
    return messages;
}

} // namespace

SecurityAdminController::SecurityAdminController(QObject* parent)
    : QObject(parent)
{
}

void SecurityAdminController::setPythonExecutable(const QString& executable)
{
    if (executable.trimmed().isEmpty()) {
        return;
    }
    m_pythonExecutable = executable;
}

void SecurityAdminController::setProfilesPath(const QString& path)
{
    m_profilesPath = bot::shell::utils::expandPath(path);
}

void SecurityAdminController::setLicensePath(const QString& path)
{
    m_licensePath = bot::shell::utils::expandPath(path);
}

void SecurityAdminController::setLogPath(const QString& path)
{
    m_logPath = bot::shell::utils::expandPath(path);
}

void SecurityAdminController::setTpmQuotePath(const QString& path)
{
    const QString normalized = bot::shell::utils::expandPath(path);
    if (m_tpmQuotePath == normalized)
        return;
    m_tpmQuotePath = normalized;
}

void SecurityAdminController::setIntegrityManifestPath(const QString& path)
{
    const QString normalized = bot::shell::utils::expandPath(path);
    if (m_integrityManifestPath == normalized)
        return;
    m_integrityManifestPath = normalized;
}

bool SecurityAdminController::refresh()
{
    if (m_busy) {
        return false;
    }
    m_busy = true;
    Q_EMIT busyChanged();

    const QByteArray overridePath = qgetenv("BOT_CORE_UI_SECURITY_STATE_PATH");
    bool ok = false;
    if (!overridePath.isEmpty()) {
        ok = loadStateFromFile(QString::fromUtf8(overridePath));
    } else {
        QStringList args;
        args << QStringLiteral("-m")
             << QStringLiteral("bot_core.security.ui_bridge")
             << QStringLiteral("dump");
        if (!m_licensePath.isEmpty()) {
            args << QStringLiteral("--license-path") << m_licensePath;
        }
        if (!m_profilesPath.isEmpty()) {
            args << QStringLiteral("--profiles-path") << m_profilesPath;
        }
        if (!m_logPath.isEmpty()) {
            args << QStringLiteral("--audit-path") << m_logPath;
        }
        args << QStringLiteral("--audit-limit") << QString::number(200);

        QByteArray stdoutData;
        QByteArray stderrData;
        ok = runBridge(args, &stdoutData, &stderrData) && loadStateFromJson(stdoutData);
        if (!stderrData.isEmpty()) {
            qCWarning(lcSecurityAdmin) << "Bridge stderr:" << QString::fromUtf8(stderrData);
        }
    }

    m_busy = false;
    Q_EMIT busyChanged();
    return ok;
}

bool SecurityAdminController::assignProfile(const QString& userId,
                                            const QStringList& roles,
                                            const QString& displayName)
{
    if (m_busy) {
        return false;
    }
    QString trimmedId = userId.trimmed();
    if (trimmedId.isEmpty()) {
        qCWarning(lcSecurityAdmin) << "Odmowa aktualizacji profilu – pusty identyfikator użytkownika";
        return false;
    }

    m_busy = true;
    Q_EMIT busyChanged();

    QStringList args;
    args << QStringLiteral("-m")
         << QStringLiteral("bot_core.security.ui_bridge")
         << QStringLiteral("assign-profile")
         << QStringLiteral("--user") << trimmedId;
    for (const QString& role : roles) {
        const QString trimmedRole = role.trimmed();
        if (!trimmedRole.isEmpty()) {
            args << QStringLiteral("--role") << trimmedRole;
        }
    }
    if (!displayName.trimmed().isEmpty()) {
        args << QStringLiteral("--display-name") << displayName.trimmed();
    }
    if (!m_profilesPath.isEmpty()) {
        args << QStringLiteral("--profiles-path") << m_profilesPath;
    }
    if (!m_logPath.isEmpty()) {
        args << QStringLiteral("--log-path") << m_logPath;
    }
    const QByteArray actor = qgetenv("BOT_CORE_UI_ADMIN_ACTOR");
    if (!actor.isEmpty()) {
        args << QStringLiteral("--actor") << QString::fromUtf8(actor);
    }

    QByteArray stdoutData;
    QByteArray stderrData;
    const bool ok = runBridge(args, &stdoutData, &stderrData);
    if (!stderrData.isEmpty()) {
        qCWarning(lcSecurityAdmin) << "Bridge stderr:" << QString::fromUtf8(stderrData);
    }
    bool shouldRefresh = false;
    if (ok) {
        QJsonParseError parseError{};
        const QJsonDocument doc = QJsonDocument::fromJson(stdoutData, &parseError);
        if (parseError.error != QJsonParseError::NoError) {
            qCWarning(lcSecurityAdmin) << "Niepoprawna odpowiedź bridge assign-profile" << parseError.errorString();
        } else if (doc.isObject()) {
            const QJsonObject obj = doc.object();
            const QString status = obj.value(QStringLiteral("status")).toString();
            if (status.compare(QStringLiteral("ok"), Qt::CaseInsensitive) == 0) {
                const QString message = QStringLiteral("Zaktualizowano profil %1").arg(trimmedId);
                Q_EMIT adminEventLogged(message);
                QVariantMap details;
                details.insert(QStringLiteral("userId"), trimmedId);
                QVariantList roleList;
                for (const QString& role : roles)
                    roleList.append(role);
                details.insert(QStringLiteral("roles"), roleList);
                recordAuditEvent(QStringLiteral("profiles"), message, details);
                shouldRefresh = true;
            } else {
                qCWarning(lcSecurityAdmin) << "Bridge zwrócił status" << status;
            }
        }
    }

    m_busy = false;
    Q_EMIT busyChanged();

    if (shouldRefresh) {
        refresh();
    }

    return ok;
}

bool SecurityAdminController::removeProfile(const QString& userId)
{
    if (m_busy) {
        return false;
    }

    const QString trimmedId = userId.trimmed();
    if (trimmedId.isEmpty()) {
        qCWarning(lcSecurityAdmin) << "Odmowa usunięcia profilu – pusty identyfikator użytkownika";
        return false;
    }

    m_busy = true;
    Q_EMIT busyChanged();

    QStringList args;
    args << QStringLiteral("-m")
         << QStringLiteral("bot_core.security.ui_bridge")
         << QStringLiteral("remove-profile")
         << QStringLiteral("--user") << trimmedId;
    if (!m_profilesPath.isEmpty()) {
        args << QStringLiteral("--profiles-path") << m_profilesPath;
    }
    if (!m_logPath.isEmpty()) {
        args << QStringLiteral("--log-path") << m_logPath;
    }
    const QByteArray actor = qgetenv("BOT_CORE_UI_ADMIN_ACTOR");
    if (!actor.isEmpty()) {
        args << QStringLiteral("--actor") << QString::fromUtf8(actor);
    }

    QByteArray stdoutData;
    QByteArray stderrData;
    bool success = runBridge(args, &stdoutData, &stderrData);
    if (!stderrData.isEmpty()) {
        qCWarning(lcSecurityAdmin) << "Bridge stderr:" << QString::fromUtf8(stderrData);
    }

    bool shouldRefresh = false;
    if (success) {
        QJsonParseError parseError{};
        const QJsonDocument doc = QJsonDocument::fromJson(stdoutData, &parseError);
        if (parseError.error != QJsonParseError::NoError) {
            qCWarning(lcSecurityAdmin) << "Niepoprawna odpowiedź bridge remove-profile" << parseError.errorString();
            success = false;
        } else if (doc.isObject()) {
            const QJsonObject obj = doc.object();
            const QString status = obj.value(QStringLiteral("status")).toString();
            if (status.compare(QStringLiteral("ok"), Qt::CaseInsensitive) == 0) {
                const QString message = QStringLiteral("Usunięto profil %1").arg(trimmedId);
                Q_EMIT adminEventLogged(message);
                QVariantMap details;
                details.insert(QStringLiteral("userId"), trimmedId);
                recordAuditEvent(QStringLiteral("profiles"), message, details);
                shouldRefresh = true;
            } else {
                qCWarning(lcSecurityAdmin) << "Bridge remove-profile zwrócił status" << status;
                success = false;
            }
        } else {
            success = false;
        }
    }

    m_busy = false;
    Q_EMIT busyChanged();

    if (shouldRefresh) {
        refresh();
    }

    return success;
}

bool SecurityAdminController::runBridge(const QStringList& arguments,
                                        QByteArray* stdoutData,
                                        QByteArray* stderrData) const
{
    QProcess process;
    process.setProgram(m_pythonExecutable);
    process.setArguments(arguments);
    process.start();
    if (!process.waitForFinished()) {
        qCWarning(lcSecurityAdmin) << "Nie udało się uruchomić bridge" << m_pythonExecutable
                                   << process.errorString();
        return false;
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        qCWarning(lcSecurityAdmin) << "Bridge zakończył się kodem" << process.exitCode();
        if (stderrData) {
            *stderrData = process.readAllStandardError();
        }
        if (stdoutData) {
            *stdoutData = process.readAllStandardOutput();
        }
        return false;
    }
    if (stdoutData) {
        *stdoutData = process.readAllStandardOutput();
    }
    if (stderrData) {
        *stderrData = process.readAllStandardError();
    }
    return true;
}

bool SecurityAdminController::loadStateFromJson(const QByteArray& data)
{
    QJsonParseError parseError{};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        qCWarning(lcSecurityAdmin) << "Nie udało się sparsować JSON bridge:" << parseError.errorString();
        return false;
    }
    const QJsonObject root = doc.object();
    const QJsonObject licenseObject = root.value(QStringLiteral("license")).toObject();
    QVariantMap license;
    for (auto it = licenseObject.begin(); it != licenseObject.end(); ++it) {
        license.insert(it.key(), it.value().toVariant());
    }
    m_licenseInfo = license;
    Q_EMIT licenseInfoChanged();

    const QVariantMap policy = license.value(QStringLiteral("policy")).toMap();
    if (!policy.isEmpty()) {
        const QString state = policy.value(QStringLiteral("state")).toString();
        if (state.compare(QStringLiteral("expired"), Qt::CaseInsensitive) == 0) {
            recordAuditEvent(QStringLiteral("license"),
                             tr("Licencja utrzymaniowa wygasła."),
                             policy);
            Q_EMIT securityAlertRaised(QStringLiteral("security:license"),
                                       2,
                                       tr("Licencja OEM"),
                                       tr("Licencja utrzymaniowa wygasła i wymaga odnowienia."));
        } else if (state.compare(QStringLiteral("critical"), Qt::CaseInsensitive) == 0
                   || state.compare(QStringLiteral("warning"), Qt::CaseInsensitive) == 0) {
            recordAuditEvent(QStringLiteral("license"),
                             tr("Licencja zbliża się do wygaśnięcia."),
                             policy);
            Q_EMIT securityAlertRaised(QStringLiteral("security:license"),
                                       1,
                                       tr("Licencja OEM"),
                                       tr("Licencja wkrótce wygaśnie – zaplanuj odnowienie."));
        }
    }

    QVariantList profiles;
    const QJsonArray profilesArray = root.value(QStringLiteral("profiles")).toArray();
    profiles.reserve(profilesArray.size());
    for (const QJsonValue& value : profilesArray) {
        if (!value.isObject()) {
            continue;
        }
        const QJsonObject obj = value.toObject();
        QVariantMap map;
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            map.insert(it.key(), it.value().toVariant());
        }
        profiles.append(map);
    }
    m_userProfiles = profiles;
    Q_EMIT userProfilesChanged();

    QVariantList auditEntries;
    const QJsonObject auditObject = root.value(QStringLiteral("audit")).toObject();
    const QJsonArray auditArray = auditObject.value(QStringLiteral("entries")).toArray();
    auditEntries.reserve(auditArray.size());
    for (const QJsonValue& entryValue : auditArray) {
        if (!entryValue.isObject())
            continue;
        auditEntries.prepend(entryValue.toObject().toVariantMap());
    }
    if (!auditEntries.isEmpty()) {
        m_auditLog = auditEntries;
        Q_EMIT auditLogChanged();
    }
    return true;
}

bool SecurityAdminController::loadStateFromFile(const QString& path)
{
    QFile file(path);
    if (!file.exists()) {
        qCWarning(lcSecurityAdmin) << "Plik zastępczego stanu bezpieczeństwa nie istnieje:" << path;
        return false;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCWarning(lcSecurityAdmin) << "Nie można otworzyć pliku" << path << file.errorString();
        return false;
    }
    const QByteArray data = file.readAll();
    return loadStateFromJson(data);
}

bool SecurityAdminController::verifyTpmBinding()
{
    QString evidencePath = m_tpmQuotePath;
    if (evidencePath.isEmpty()) {
        const QByteArray envPath = qgetenv("BOT_CORE_UI_TPM_QUOTE");
        if (!envPath.isEmpty())
            evidencePath = bot::shell::utils::expandPath(QString::fromUtf8(envPath));
    }

    QVariantMap status;
    status.insert(QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    if (evidencePath.isEmpty()) {
        status.insert(QStringLiteral("error"), tr("Brak ścieżki do pliku z dowodem TPM."));
        status.insert(QStringLiteral("valid"), false);
        m_tpmStatus = status;
        Q_EMIT tpmStatusChanged();
        recordAuditEvent(QStringLiteral("tpm"), tr("Niepowodzenie walidacji TPM"), status);
        Q_EMIT securityAlertRaised(QStringLiteral("security:tpm"), 2, tr("Weryfikacja TPM"),
                                   tr("Nie znaleziono dowodu TPM."));
        return false;
    }

    status.insert(QStringLiteral("evidencePath"), evidencePath);
    const QVariantMap evidence = readTpmEvidence(evidencePath);
    status.insert(QStringLiteral("evidence"), evidence);

    const QVariantMap fingerprintInfo = m_licenseInfo.value(QStringLiteral("fingerprint")).toMap();
    const QString expectedFingerprint = fingerprintInfo.value(QStringLiteral("hash")).toString();
    status.insert(QStringLiteral("expectedFingerprint"), expectedFingerprint);
    const QString sealedFingerprint = evidence.value(QStringLiteral("sealed_fingerprint")).toString();
    status.insert(QStringLiteral("sealedFingerprint"), sealedFingerprint);

    const bool valid = !expectedFingerprint.isEmpty() && !sealedFingerprint.isEmpty()
        && sealedFingerprint.startsWith(expectedFingerprint.left(16));
    status.insert(QStringLiteral("valid"), valid);

    const bool changed = (status != m_tpmStatus);
    m_tpmStatus = status;
    if (changed)
        Q_EMIT tpmStatusChanged();

    recordAuditEvent(QStringLiteral("tpm"),
                     valid ? tr("Weryfikacja TPM zakończona sukcesem")
                           : tr("Weryfikacja TPM nie powiodła się"),
                     status);

    if (!valid) {
        const QString detail = sealedFingerprint.isEmpty()
            ? tr("Dowód TPM nie zawiera fingerprintu.")
            : tr("Fingerprint z TPM nie zgadza się z licencją.");
        Q_EMIT securityAlertRaised(QStringLiteral("security:tpm"),
                                   2,
                                   tr("Weryfikacja TPM"),
                                   detail);
    }

    return valid;
}

bool SecurityAdminController::runIntegrityCheck()
{
    QVariantMap report;
    const bool ok = evaluateIntegrityManifest(&report);
    const bool changed = (report != m_lastIntegrityReport);
    m_lastIntegrityReport = report;
    if (changed)
        Q_EMIT integrityReportChanged();

    recordAuditEvent(QStringLiteral("integrity"),
                     ok ? tr("Kontrola integralności zakończona sukcesem")
                        : tr("Kontrola integralności wykryła odchylenia"),
                     report);

    const QVariantList mismatches = report.value(QStringLiteral("mismatches")).toList();
    if (!ok || !mismatches.isEmpty()) {
        const QString detail = mismatches.isEmpty()
            ? tr("Sprawdź logi integralności po szczegóły.")
            : tr("Wykryto %1 niespójności plików.").arg(mismatches.size());
        Q_EMIT securityAlertRaised(QStringLiteral("security:integrity"), 1, tr("Kontrola integralności"), detail);
    }

    return ok;
}

QVariantMap SecurityAdminController::readTpmEvidence(const QString& path) const
{
    QVariantMap evidence;
    QFile file(path);
    if (!file.exists())
        return evidence;
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return evidence;

    const QByteArray payload = file.readAll();
    file.close();
    QJsonParseError parseError{};
    const QJsonDocument doc = QJsonDocument::fromJson(payload, &parseError);
    if (parseError.error == QJsonParseError::NoError && doc.isObject())
        evidence = doc.object().toVariantMap();
    else
        evidence.insert(QStringLiteral("raw"), QString::fromUtf8(payload).trimmed());
    return evidence;
}

void SecurityAdminController::recordAuditEvent(const QString& category,
                                               const QString& message,
                                               const QVariantMap& details)
{
    QVariantMap entry;
    entry.insert(QStringLiteral("timestamp"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    entry.insert(QStringLiteral("category"), category);
    entry.insert(QStringLiteral("message"), message);
    if (!details.isEmpty())
        entry.insert(QStringLiteral("details"), details);

    m_auditLog.prepend(entry);
    constexpr int kMaxAuditEntries = 200;
    while (m_auditLog.size() > kMaxAuditEntries)
        m_auditLog.removeLast();

    Q_EMIT auditLogChanged();
}

QString SecurityAdminController::computeFileDigest(const QString& path) const
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
        return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd())
        hash.addData(file.read(8192));
    return QString::fromLatin1(hash.result().toHex());
}

bool SecurityAdminController::evaluateIntegrityManifest(QVariantMap* report)
{
    QString manifestPath = m_integrityManifestPath;
    if (manifestPath.isEmpty()) {
        const QByteArray envPath = qgetenv("BOT_CORE_UI_INTEGRITY_MANIFEST");
        if (!envPath.isEmpty())
            manifestPath = bot::shell::utils::expandPath(QString::fromUtf8(envPath));
    }
    if (manifestPath.isEmpty())
        manifestPath = bot::shell::utils::expandPath(QStringLiteral("config/integrity_manifest.json"));

    QVariantMap localReport;
    localReport.insert(QStringLiteral("manifestPath"), manifestPath);
    localReport.insert(QStringLiteral("checkedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs));
    QVariantList mismatches;

    if (manifestPath.isEmpty()) {
        localReport.insert(QStringLiteral("valid"), false);
        localReport.insert(QStringLiteral("error"), tr("Nie wskazano manifestu integralności."));
        if (report)
            *report = localReport;
        return false;
    }

    QFile manifest(manifestPath);
    if (!manifest.open(QIODevice::ReadOnly | QIODevice::Text)) {
        localReport.insert(QStringLiteral("valid"), false);
        localReport.insert(QStringLiteral("error"), tr("Nie można otworzyć manifestu %1").arg(manifestPath));
        if (report)
            *report = localReport;
        return false;
    }

    const QByteArray data = manifest.readAll();
    manifest.close();
    QJsonParseError parseError{};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        localReport.insert(QStringLiteral("valid"), false);
        localReport.insert(QStringLiteral("error"), tr("Manifest ma niepoprawny format (%1)").arg(parseError.errorString()));
        if (report)
            *report = localReport;
        return false;
    }

    const QJsonArray filesArray = doc.object().value(QStringLiteral("files")).toArray();
    for (const QJsonValue& value : filesArray) {
        if (!value.isObject())
            continue;
        const QJsonObject obj = value.toObject();
        const QString relativePath = obj.value(QStringLiteral("path")).toString();
        const QString expectedDigest = obj.value(QStringLiteral("sha256")).toString();
        const QString absolutePath = bot::shell::utils::expandPath(relativePath);
        QString actualDigest = computeFileDigest(absolutePath);
        if (actualDigest.isEmpty()) {
            QVariantMap mismatch;
            mismatch.insert(QStringLiteral("path"), absolutePath);
            mismatch.insert(QStringLiteral("reason"), tr("Nie można odczytać pliku."));
            mismatches.append(mismatch);
            continue;
        }
        if (!expectedDigest.isEmpty() && actualDigest.compare(expectedDigest, Qt::CaseInsensitive) != 0) {
            QVariantMap mismatch;
            mismatch.insert(QStringLiteral("path"), absolutePath);
            mismatch.insert(QStringLiteral("expected"), expectedDigest);
            mismatch.insert(QStringLiteral("actual"), actualDigest);
            mismatches.append(mismatch);
        }
    }

    const bool valid = mismatches.isEmpty();
    localReport.insert(QStringLiteral("valid"), valid);
    localReport.insert(QStringLiteral("mismatches"), mismatches);
    if (report)
        *report = localReport;
    return valid;
}

