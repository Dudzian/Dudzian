#include "OfflineUpdateManager.hpp"

#include <QByteArray>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QDirIterator>
#include <QObject>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QStringList>
#include <QProcess>
#include <QMetaType>
#include <QSaveFile>
#include <QtGlobal>

#include "license/LicenseActivationController.hpp"

#ifdef HAS_LIBARCHIVE_FALLBACK
#include <archive.h>
#include <archive_entry.h>
#endif

Q_LOGGING_CATEGORY(lcOfflineUpdates, "bot.shell.update.offline")

namespace {
constexpr auto kManifestFileName = "manifest.json";
constexpr auto kPayloadFileName = "payload.tar";
constexpr auto kPatchFileName = "patch.tar";
constexpr auto kStagingDirectory = ".staging";

QString normalizeForProcess(const QString& path)
{
#if defined(Q_OS_WIN)
    return QDir::toNativeSeparators(path);
#else
    return path;
#endif
}

bool runArchiveCommand(const QStringList& arguments, QString* errorMessage)
{
    const QStringList candidates{QStringLiteral("tar"), QStringLiteral("bsdtar")};
    for (const QString& program : candidates) {
        QProcess process;
        process.setProgram(program);
        process.setArguments(arguments);
        process.start();
        if (!process.waitForStarted())
            continue;
        process.closeWriteChannel();
        if (!process.waitForFinished(-1))
            continue;
        if (process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0)
            return true;
        qCWarning(lcOfflineUpdates) << "Proces archiwizujący zakończył się błędem" << program
                                    << process.exitCode() << process.readAllStandardError();
    }
    if (errorMessage)
        *errorMessage = QObject::tr("Brak narzędzia archiwizującego (tar/bsdtar).");
    return false;
}

QString computeFileSha256(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
        return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        hash.addData(file.read(4096));
    }
    return QString::fromLatin1(hash.result().toHex());
}

#ifdef HAS_LIBARCHIVE_FALLBACK

constexpr int kLibArchiveBlockSize = 16384;
const QFile::Permissions kDefaultFilePermissions = QFile::ReadOwner | QFile::WriteOwner |
                                                   QFile::ReadGroup | QFile::ReadOther;
const QFile::Permissions kDefaultDirPermissions = kDefaultFilePermissions |
                                                  QFile::ExeOwner | QFile::ExeGroup |
                                                  QFile::ExeOther;

bool extractArchiveWithLibArchive(
    const QString& archivePath, const QString& targetDir, QString* errorMessage
)
{
    auto setError = [&](const QString& message) {
        if (errorMessage && errorMessage->isEmpty())
            *errorMessage = message;
    };

    struct archive* reader = archive_read_new();
    if (reader == nullptr) {
        setError(QObject::tr("Nie można zainicjalizować libarchive."));
        return false;
    }

    archive_read_support_filter_all(reader);
    archive_read_support_format_tar(reader);

    const QByteArray encoded = QFile::encodeName(archivePath);
    if (archive_read_open_filename(reader, encoded.constData(), kLibArchiveBlockSize) != ARCHIVE_OK) {
        setError(QObject::tr("Nie można otworzyć archiwum %1 (libarchive).")
                      .arg(archivePath));
        archive_read_free(reader);
        return false;
    }

    bool success = true;
    struct archive_entry* entry = nullptr;
    while (success && archive_read_next_header(reader, &entry) == ARCHIVE_OK) {
        const QString relativePath = QString::fromUtf8(archive_entry_pathname(entry));
        const QString destination = QDir(targetDir).filePath(relativePath);

        switch (archive_entry_filetype(entry)) {
        case AE_IFDIR: {
            if (!QDir().mkpath(destination)) {
                setError(QObject::tr("Nie można utworzyć katalogu %1.").arg(destination));
                success = false;
            } else {
                QFile::setPermissions(destination, kDefaultDirPermissions);
            }
            break;
        }
        case AE_IFLNK: {
            const char* linkTarget = archive_entry_symlink(entry);
            if (linkTarget == nullptr) {
                setError(QObject::tr("Niepoprawne dowiązanie symboliczne w archiwum: %1")
                              .arg(relativePath));
                success = false;
                break;
            }
            QFile::remove(destination);
            if (!QFile::link(QString::fromUtf8(linkTarget), destination)) {
                setError(QObject::tr("Nie można utworzyć dowiązania %1.").arg(destination));
                success = false;
            }
            break;
        }
        default: {
            const QFileInfo info(destination);
            QDir parent = info.dir();
            if (!parent.exists() && !parent.mkpath(QStringLiteral("."))) {
                setError(QObject::tr("Nie można utworzyć katalogu %1.").arg(parent.absolutePath()));
                success = false;
                break;
            }

            QFile file(destination);
            if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                setError(QObject::tr("Nie można zapisać pliku %1.").arg(destination));
                success = false;
                break;
            }

            QByteArray buffer(kLibArchiveBlockSize, Qt::Uninitialized);
            la_ssize_t len = 0;
            while (success && (len = archive_read_data(reader, buffer.data(), buffer.size())) > 0) {
                if (file.write(buffer.constData(), len) != len) {
                    setError(QObject::tr("Nie można zapisać danych do %1.").arg(destination));
                    success = false;
                }
            }
            if (success && len < 0) {
                setError(QObject::tr("Błąd libarchive podczas odczytu %1: %2")
                              .arg(relativePath, QString::fromUtf8(archive_error_string(reader) ?: "")));
                success = false;
            }
            file.close();
            if (success)
                file.setPermissions(kDefaultFilePermissions);
            break;
        }
        }
    }

    archive_read_close(reader);
    archive_read_free(reader);
    return success;
}

bool addEntryToArchive(
    struct archive* writer,
    const QString& relativePath,
    const QFileInfo& info,
    QString* errorMessage
)
{
    struct archive_entry* entry = archive_entry_new();
    if (entry == nullptr) {
        if (errorMessage)
            *errorMessage = QObject::tr("Nie można zaalokować wpisu archiwum dla %1.").arg(relativePath);
        return false;
    }
    archive_entry_set_pathname(entry, relativePath.toUtf8().constData());

    if (info.isDir()) {
        archive_entry_set_filetype(entry, AE_IFDIR);
        archive_entry_set_perm(entry, 0755);
        archive_entry_set_size(entry, 0);
        const int result = archive_write_header(writer, entry);
        archive_entry_free(entry);
        if (result != ARCHIVE_OK) {
            if (errorMessage)
                *errorMessage = QObject::tr("Nie można dodać katalogu %1 do archiwum.").arg(relativePath);
            return false;
        }
        return true;
    }

    if (info.isSymLink()) {
        archive_entry_set_filetype(entry, AE_IFLNK);
        archive_entry_set_perm(entry, 0755);
        archive_entry_set_size(entry, 0);
        const QByteArray linkTarget = info.symLinkTarget().toUtf8();
        archive_entry_set_symlink(entry, linkTarget.constData());
        const int result = archive_write_header(writer, entry);
        archive_entry_free(entry);
        if (result != ARCHIVE_OK) {
            if (errorMessage)
                *errorMessage = QObject::tr("Nie można dodać dowiązania %1 do archiwum.").arg(relativePath);
            return false;
        }
        return true;
    }

    archive_entry_set_filetype(entry, AE_IFREG);
    archive_entry_set_perm(entry, 0644);
    archive_entry_set_size(entry, info.size());

    if (archive_write_header(writer, entry) != ARCHIVE_OK) {
        archive_entry_free(entry);
        if (errorMessage)
            *errorMessage = QObject::tr("Nie można dodać pliku %1 do archiwum.").arg(relativePath);
        return false;
    }

    QFile file(info.filePath());
    if (!file.open(QIODevice::ReadOnly)) {
        if (errorMessage)
            *errorMessage = QObject::tr("Nie można odczytać pliku %1.").arg(info.filePath());
        archive_entry_free(entry);
        return false;
    }

    QByteArray buffer(kLibArchiveBlockSize, Qt::Uninitialized);
    while (!file.atEnd()) {
        const qint64 readBytes = file.read(buffer.data(), buffer.size());
        if (readBytes < 0) {
            if (errorMessage)
                *errorMessage = QObject::tr("Błąd odczytu pliku %1.").arg(info.filePath());
            archive_entry_free(entry);
            return false;
        }
        if (readBytes == 0)
            break;
        if (archive_write_data(writer, buffer.constData(), static_cast<size_t>(readBytes)) < 0) {
            if (errorMessage)
                *errorMessage = QObject::tr("Nie można zapisać danych archiwum dla %1.").arg(relativePath);
            archive_entry_free(entry);
            return false;
        }
    }
    archive_entry_free(entry);
    return true;
}

bool createArchiveWithLibArchive(
    const QString& sourceDir, const QString& archivePath, QString* errorMessage
)
{
    struct archive* writer = archive_write_new();
    if (writer == nullptr) {
        if (errorMessage)
            *errorMessage = QObject::tr("Nie można zainicjalizować libarchive do zapisu.");
        return false;
    }

    archive_write_set_format_pax_restricted(writer);

    const QByteArray encoded = QFile::encodeName(archivePath);
    if (archive_write_open_filename(writer, encoded.constData()) != ARCHIVE_OK) {
        if (errorMessage)
            *errorMessage = QObject::tr("Nie można otworzyć docelowego archiwum %1.").arg(archivePath);
        archive_write_free(writer);
        return false;
    }

    bool success = true;
    const QDir rootDir(sourceDir);
    QDirIterator it(
        sourceDir,
        QDir::NoDotAndDotDot | QDir::AllEntries | QDir::Hidden,
        QDirIterator::Subdirectories
    );

    while (success && it.hasNext()) {
        const QString path = it.next();
        const QFileInfo info = it.fileInfo();
        const QString relative = rootDir.relativeFilePath(path);
        success = addEntryToArchive(writer, relative, info, errorMessage);
    }

    archive_write_close(writer);
    archive_write_free(writer);
    if (!success && errorMessage && errorMessage->isEmpty())
        *errorMessage = QObject::tr("Nie udało się utworzyć archiwum %1.").arg(archivePath);
    return success;
}

#else

bool extractArchiveWithLibArchive(const QString& archivePath, const QString& targetDir, QString* errorMessage)
{
    Q_UNUSED(archivePath)
    Q_UNUSED(targetDir)
    Q_UNUSED(errorMessage)
    return false;
}

bool createArchiveWithLibArchive(
    const QString& sourceDir, const QString& archivePath, QString* errorMessage
)
{
    Q_UNUSED(sourceDir)
    Q_UNUSED(archivePath)
    Q_UNUSED(errorMessage)
    return false;
}

#endif
}

OfflineUpdateManager::OfflineUpdateManager(QObject* parent)
    : QObject(parent)
{
}

void OfflineUpdateManager::setPackagesDirectory(const QString& path)
{
    const QString normalized = QDir::cleanPath(QDir::current().absoluteFilePath(path));
    if (m_packagesDir == normalized)
        return;
    m_packagesDir = normalized;
}

void OfflineUpdateManager::setInstallDirectory(const QString& path)
{
    const QString normalized = QDir::cleanPath(QDir::current().absoluteFilePath(path));
    if (m_installDir == normalized)
        return;
    m_installDir = normalized;
}

void OfflineUpdateManager::setStateFilePath(const QString& path)
{
    const QString normalized = QDir::cleanPath(QDir::current().absoluteFilePath(path));
    if (m_stateFile == normalized)
        return;
    m_stateFile = normalized;
}

void OfflineUpdateManager::setLicenseController(LicenseActivationController* controller)
{
    if (m_licenseController == controller)
        return;
    m_licenseController = controller;
}

void OfflineUpdateManager::setFingerprintOverride(const QString& fingerprint)
{
    if (m_fingerprintOverride == fingerprint)
        return;
    m_fingerprintOverride = fingerprint;
}

void OfflineUpdateManager::setTpmEvidencePath(const QString& path)
{
    const QString normalized = QDir::cleanPath(QDir::current().absoluteFilePath(path));
    if (m_tpmEvidencePath == normalized)
        return;
    m_tpmEvidencePath = normalized;
}

bool OfflineUpdateManager::refresh()
{
    if (m_packagesDir.isEmpty()) {
        m_lastError = tr("Nie skonfigurowano katalogu pakietów aktualizacji.");
        Q_EMIT lastErrorChanged();
        return false;
    }

    QString errorMessage;
    const QList<UpdatePackage> packages = loadPackages(&errorMessage);
    if (!errorMessage.isEmpty()) {
        m_lastError = errorMessage;
        Q_EMIT lastErrorChanged();
        return false;
    }

    QVariantList available;
    for (const UpdatePackage& pkg : packages) {
        QVariantMap item = pkg.metadata;
        item.insert(QStringLiteral("id"), pkg.id);
        item.insert(QStringLiteral("version"), pkg.version);
        item.insert(QStringLiteral("fingerprint"), pkg.fingerprint);
        item.insert(QStringLiteral("differential"), pkg.differential);
        if (!pkg.baseId.isEmpty())
            item.insert(QStringLiteral("baseId"), pkg.baseId);
        available.append(item);
    }

    if (available != m_availableUpdates) {
        m_availableUpdates = available;
        Q_EMIT availableUpdatesChanged();
    }

    if (!loadState())
        qCWarning(lcOfflineUpdates) << "Nie udało się załadować stanu aktualizacji" << m_stateFile;

    return true;
}

bool OfflineUpdateManager::applyUpdate(const QString& packageId)
{
    if (packageId.trimmed().isEmpty())
        return false;

    const UpdatePackage pkg = packageById(packageId);
    if (pkg.id.isEmpty()) {
        m_lastError = tr("Pakiet %1 nie został znaleziony.").arg(packageId);
        Q_EMIT lastErrorChanged();
        return false;
    }

    QString message;
    if (!verifyPackageSignature(pkg, &message) || !verifyFingerprint(pkg, &message)) {
        m_lastError = message;
        Q_EMIT lastErrorChanged();
        Q_EMIT updateFailed(pkg.id, message);
        return false;
    }

    if (!pkg.differential) {
        if (!applyPackage(pkg)) {
            m_lastError = tr("Nie udało się zainstalować pakietu %1.").arg(pkg.id);
            Q_EMIT lastErrorChanged();
            Q_EMIT updateFailed(pkg.id, m_lastError);
            return false;
        }
    } else {
        const UpdatePackage basePkg = packageById(pkg.baseId);
        if (basePkg.id.isEmpty()) {
            m_lastError = tr("Pakiet bazowy %1 dla łatki %2 nie istnieje.").arg(pkg.baseId, pkg.id);
            Q_EMIT lastErrorChanged();
            Q_EMIT updateFailed(pkg.id, m_lastError);
            return false;
        }
        if (!applyDifferential(basePkg, pkg)) {
            m_lastError = tr("Nie udało się zastosować różnicowej łatki %1.").arg(pkg.id);
            Q_EMIT lastErrorChanged();
            Q_EMIT updateFailed(pkg.id, m_lastError);
            return false;
        }
    }

    if (!m_lastError.isEmpty()) {
        m_lastError.clear();
        Q_EMIT lastErrorChanged();
    }

    Q_EMIT updateCompleted(pkg.id);
    return refresh();
}

bool OfflineUpdateManager::rollbackUpdate(const QString& packageId)
{
    Q_UNUSED(packageId);
    if (m_installDir.isEmpty()) {
        m_lastError = tr("Brak katalogu instalacyjnego.");
        Q_EMIT lastErrorChanged();
        return false;
    }
    // Rollback jest symulowany poprzez usunięcie wpisu ze stanu
    QStringList ids = installedIds();
    if (!ids.removeOne(packageId)) {
        m_lastError = tr("Pakiet %1 nie jest zainstalowany.").arg(packageId);
        Q_EMIT lastErrorChanged();
        return false;
    }
    m_installedUpdates.clear();
    for (const QString& id : std::as_const(ids)) {
        QVariantMap item;
        item.insert(QStringLiteral("id"), id);
        item.insert(QStringLiteral("rolledBackAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
        m_installedUpdates.append(item);
    }
    persistState();
    Q_EMIT installedUpdatesChanged();
    return true;
}

bool OfflineUpdateManager::applyDifferentialPatch(const QString& baseId, const QString& patchId)
{
    const UpdatePackage basePkg = packageById(baseId);
    const UpdatePackage patchPkg = packageById(patchId);
    if (basePkg.id.isEmpty() || patchPkg.id.isEmpty()) {
        m_lastError = tr("Nie znaleziono pakietów %1/%2.").arg(baseId, patchId);
        Q_EMIT lastErrorChanged();
        return false;
    }
    if (!patchPkg.differential) {
        m_lastError = tr("Pakiet %1 nie jest łatką różnicową.").arg(patchId);
        Q_EMIT lastErrorChanged();
        return false;
    }
    if (!applyDifferential(basePkg, patchPkg)) {
        m_lastError = tr("Nie udało się zastosować łatki różnicowej.");
        Q_EMIT lastErrorChanged();
        return false;
    }
    return refresh();
}

QVariantMap OfflineUpdateManager::describeUpdate(const QString& packageId) const
{
    const UpdatePackage pkg = packageById(packageId);
    QVariantMap map;
    if (pkg.id.isEmpty())
        return map;
    map = pkg.metadata;
    map.insert(QStringLiteral("id"), pkg.id);
    map.insert(QStringLiteral("version"), pkg.version);
    map.insert(QStringLiteral("fingerprint"), pkg.fingerprint);
    map.insert(QStringLiteral("differential"), pkg.differential);
    if (!pkg.baseId.isEmpty())
        map.insert(QStringLiteral("baseId"), pkg.baseId);
    return map;
}

void OfflineUpdateManager::setBusy(bool busy)
{
    if (m_busy == busy)
        return;
    m_busy = busy;
    Q_EMIT busyChanged();
}

bool OfflineUpdateManager::verifyPackageSignature(const UpdatePackage& pkg, QString* message) const
{
    QProcess process;
    QStringList arguments;
    const QString python = QStringLiteral("python3");
    const QString scriptPath = QDir::current().absoluteFilePath(QStringLiteral("scripts/update_package.py"));
    arguments << scriptPath << QStringLiteral("verify") << QStringLiteral("--package-dir") << pkg.path;
    const QByteArray keyEnv = qgetenv("BOT_CORE_UPDATE_HMAC_KEY");
    if (!keyEnv.isEmpty())
        arguments << QStringLiteral("--key") << QString::fromUtf8(keyEnv);
    process.setProgram(python);
    process.setArguments(arguments);
    process.start();
    if (!process.waitForFinished(-1)) {
        if (message)
            *message = tr("Nie udało się uruchomić weryfikatora pakietów.");
        qCWarning(lcOfflineUpdates) << "Weryfikacja pakietu" << pkg.id << "przerwana" << process.errorString();
        return false;
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        if (message)
            *message = tr("Weryfikator pakietu zwrócił błąd (%1).").arg(process.exitCode());
        qCWarning(lcOfflineUpdates) << "Weryfikacja pakietu" << pkg.id << "błąd"
                                   << QString::fromUtf8(process.readAllStandardError());
        return false;
    }

    const QByteArray stdoutData = process.readAllStandardOutput();
    QJsonParseError parseError{};
    const QJsonDocument doc = QJsonDocument::fromJson(stdoutData, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        if (message)
            *message = tr("Niepoprawna odpowiedź weryfikatora pakietu.");
        qCWarning(lcOfflineUpdates) << "Weryfikator zwrócił niepoprawny JSON" << parseError.errorString();
        return false;
    }
    const QJsonObject result = doc.object();
    const QString status = result.value(QStringLiteral("status")).toString();
    if (status.compare(QStringLiteral("ok"), Qt::CaseInsensitive) != 0) {
        if (message)
            *message = result.value(QStringLiteral("error")).toString(tr("Weryfikacja pakietu nie powiodła się."));
        return false;
    }
    return true;
}

bool OfflineUpdateManager::verifyFingerprint(const UpdatePackage& pkg, QString* message) const
{
    QString fingerprint = m_fingerprintOverride;
    if (fingerprint.trimmed().isEmpty() && m_licenseController)
        fingerprint = m_licenseController->fingerprint().value(QStringLiteral("hash")).toString();

    if (fingerprint.trimmed().isEmpty()) {
        if (message)
            *message = tr("Brak fingerprintu urządzenia.");
        return false;
    }

    if (!pkg.fingerprint.isEmpty() && pkg.fingerprint != fingerprint) {
        if (message)
            *message = tr("Pakiet %1 jest przypisany do innego urządzenia.").arg(pkg.id);
        return false;
    }

    if (!m_tpmEvidencePath.isEmpty()) {
        QProcess process;
        QStringList args;
        args << QStringLiteral("-m")
             << QStringLiteral("bot_core.security.ui_bridge")
             << QStringLiteral("verify-tpm")
             << QStringLiteral("--evidence-path") << m_tpmEvidencePath
             << QStringLiteral("--expected-fingerprint") << fingerprint;
        process.setProgram(QStringLiteral("python3"));
        process.setArguments(args);
        process.start();
        if (!process.waitForFinished(-1) || process.exitStatus() != QProcess::NormalExit
            || process.exitCode() != 0) {
            if (message)
                *message = tr("Weryfikacja TPM dla aktualizacji nie powiodła się.");
            return false;
        }
        QJsonParseError parseError{};
        const QJsonDocument doc = QJsonDocument::fromJson(process.readAllStandardOutput(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
            if (message)
                *message = tr("Weryfikator TPM zwrócił niepoprawny JSON.");
            return false;
        }
        const QJsonObject result = doc.object();
        const QString status = result.value(QStringLiteral("status")).toString();
        const bool valid = status.compare(QStringLiteral("ok"), Qt::CaseInsensitive) == 0
            && result.value(QStringLiteral("errors")).toArray().isEmpty();
        if (!valid) {
            if (message)
                *message = tr("Dowód TPM nie pasuje do fingerprintu urządzenia.");
            return false;
        }
    }

    return true;
}

bool OfflineUpdateManager::loadState()
{
    if (m_stateFile.isEmpty())
        return true;

    QFile file(m_stateFile);
    if (!file.exists())
        return true;
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;

    const QByteArray payload = file.readAll();
    file.close();

    const QJsonDocument doc = QJsonDocument::fromJson(payload);
    if (!doc.isObject())
        return false;

    const QJsonArray installedArray = doc.object().value(QStringLiteral("installed")).toArray();
    QVariantList installedList;
    installedList.reserve(installedArray.size());
    for (const QJsonValue& value : installedArray) {
        if (value.isObject())
            installedList.append(value.toObject().toVariantMap());
    }
    if (installedList != m_installedUpdates) {
        m_installedUpdates = installedList;
        Q_EMIT installedUpdatesChanged();
    }
    return true;
}

bool OfflineUpdateManager::persistState() const
{
    if (m_stateFile.isEmpty())
        return true;

    QJsonObject root;
    QJsonArray installedArray;
    for (const QVariant& value : m_installedUpdates)
        installedArray.append(QJsonObject::fromVariantMap(value.toMap()));
    root.insert(QStringLiteral("installed"), installedArray);

    QSaveFile file(m_stateFile);
    file.setDirectWriteFallback(true);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return false;

    const QByteArray payload = QJsonDocument(root).toJson(QJsonDocument::Compact);
    if (file.write(payload) != payload.size())
        return false;
    return file.commit();
}

QList<OfflineUpdateManager::UpdatePackage> OfflineUpdateManager::loadPackages(QString* errorMessage) const
{
    QList<UpdatePackage> packages;
    if (m_packagesDir.trimmed().isEmpty()) {
        if (errorMessage)
            *errorMessage = tr("Nie skonfigurowano katalogu pakietów aktualizacji.");
        return packages;
    }

    QDir dir(m_packagesDir);
    if (!dir.exists()) {
        if (errorMessage)
            *errorMessage = tr("Katalog aktualizacji %1 nie istnieje.").arg(m_packagesDir);
        return packages;
    }

    QProcess process;
    QStringList arguments;
    const QString python = QStringLiteral("python3");
    const QString scriptPath = QDir::current().absoluteFilePath(QStringLiteral("scripts/update_package.py"));
    arguments << scriptPath << QStringLiteral("scan") << QStringLiteral("--packages-dir") << dir.absolutePath();
    process.setProgram(python);
    process.setArguments(arguments);
    process.start();
    if (!process.waitForFinished(-1) || process.exitStatus() != QProcess::NormalExit) {
        if (errorMessage)
            *errorMessage = tr("Nie udało się odczytać listy pakietów aktualizacji.");
        qCWarning(lcOfflineUpdates) << "Nie można uzyskać listy pakietów" << process.errorString();
        return packages;
    }
    if (process.exitCode() != 0) {
        if (errorMessage)
            *errorMessage = tr("Skrypt opisujący pakiety zwrócił błąd (%1).").arg(process.exitCode());
        qCWarning(lcOfflineUpdates) << "Skrypt skanowania pakietów zakończył się kodem"
                                   << process.exitCode() << process.readAllStandardError();
        return packages;
    }

    const QByteArray stdoutData = process.readAllStandardOutput();
    QJsonParseError parseError{};
    const QJsonDocument doc = QJsonDocument::fromJson(stdoutData, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        if (errorMessage)
            *errorMessage = tr("Skrypt opisujący pakiety zwrócił niepoprawny JSON.");
        qCWarning(lcOfflineUpdates) << "Niepoprawny JSON z update_package.py scan"
                                   << parseError.errorString();
        return packages;
    }

    const QJsonArray entries = doc.object().value(QStringLiteral("packages")).toArray();
    for (const QJsonValue& entryValue : entries) {
        if (!entryValue.isObject())
            continue;
        const QJsonObject obj = entryValue.toObject();
        const QString status = obj.value(QStringLiteral("status")).toString(QStringLiteral("ok"));
        if (status.compare(QStringLiteral("ok"), Qt::CaseInsensitive) != 0) {
            qCWarning(lcOfflineUpdates) << "Pakiet pominięty podczas skanowania" << obj.value(QStringLiteral("path"))
                                       << obj.value(QStringLiteral("error"));
            continue;
        }

        UpdatePackage pkg;
        pkg.path = obj.value(QStringLiteral("path")).toString();
        if (pkg.path.isEmpty())
            pkg.path = dir.absoluteFilePath(obj.value(QStringLiteral("id")).toString());
        pkg.id = obj.value(QStringLiteral("id")).toString(QFileInfo(pkg.path).fileName());
        pkg.version = obj.value(QStringLiteral("version")).toString();
        pkg.fingerprint = obj.value(QStringLiteral("fingerprint")).toString();
        pkg.signature = obj.value(QStringLiteral("signature")).toString();
        pkg.signatureObject = obj.value(QStringLiteral("signature_object")).toVariant().toMap();
        pkg.differential = obj.value(QStringLiteral("differential")).toBool();
        pkg.baseId = obj.value(QStringLiteral("base_id")).toString();
        pkg.payloadFile = obj.value(QStringLiteral("payload_file")).toString(QString::fromLatin1(kPayloadFileName));
        pkg.diffFile = obj.value(QStringLiteral("diff_file")).toString();
        pkg.integrity = obj.value(QStringLiteral("integrity")).toVariant().toMap();
        pkg.metadata = obj.value(QStringLiteral("metadata")).toVariant().toMap();
        packages.append(pkg);
    }

    return packages;
}

bool OfflineUpdateManager::applyPackage(const UpdatePackage& pkg)
{
    QString message;
    if (!copyPackagePayload(pkg, &message)) {
        qCWarning(lcOfflineUpdates) << "Nie można skopiować pakietu" << pkg.id << message;
        return false;
    }

    QVariantMap record;
    record.insert(QStringLiteral("id"), pkg.id);
    record.insert(QStringLiteral("version"), pkg.version);
    record.insert(QStringLiteral("installedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
    record.insert(QStringLiteral("metadata"), pkg.metadata);

    bool updated = false;
    for (int i = 0; i < m_installedUpdates.size(); ++i) {
        QVariantMap entry = m_installedUpdates.at(i).toMap();
        if (entry.value(QStringLiteral("id")).toString() == pkg.id) {
            m_installedUpdates[i] = record;
            updated = true;
            break;
        }
    }
    if (!updated)
        m_installedUpdates.append(record);

    persistState();
    Q_EMIT installedUpdatesChanged();
    return true;
}

bool OfflineUpdateManager::copyPackagePayload(const UpdatePackage& pkg, QString* errorMessage)
{
    if (m_installDir.isEmpty()) {
        if (errorMessage)
            *errorMessage = tr("Brak katalogu instalacyjnego.");
        return false;
    }

    QDir installDir(m_installDir);
    if (!installDir.exists() && !installDir.mkpath(QStringLiteral("."))) {
        if (errorMessage)
            *errorMessage = tr("Nie można utworzyć katalogu instalacyjnego %1.").arg(m_installDir);
        return false;
    }

    const QString payloadPath = packagePayloadPath(pkg);
    QFile payload(payloadPath);
    if (!payload.exists()) {
        if (errorMessage)
            *errorMessage = tr("Pakiet %1 nie zawiera wymaganego pliku aktualizacji.").arg(pkg.id);
        return false;
    }

    const QString targetPath = storedPayloadPath(pkg);

    if (!payload.open(QIODevice::ReadOnly)) {
        if (errorMessage)
            *errorMessage = tr("Nie udało się otworzyć pakietu %1.").arg(pkg.id);
        return false;
    }

    QSaveFile target(targetPath);
    target.setDirectWriteFallback(true);
    if (!target.open(QIODevice::WriteOnly)) {
        if (errorMessage)
            *errorMessage = tr("Nie udało się zapisać pliku %1.").arg(targetPath);
        return false;
    }

    setBusy(true);
    qint64 totalWritten = 0;
    const qint64 totalSize = payload.size();
    while (!payload.atEnd()) {
        const QByteArray chunk = payload.read(8192);
        totalWritten += target.write(chunk);
        const double progress = totalSize > 0 ? static_cast<double>(totalWritten) / static_cast<double>(totalSize)
                      : 1.0;
        Q_EMIT updateProgress(pkg.id, progress);
    }
    payload.close();

    if (!target.commit()) {
        if (errorMessage)
            *errorMessage = tr("Nie udało się zatwierdzić pliku aktualizacji.");
        setBusy(false);
        return false;
    }

    setBusy(false);
    return true;
}


bool OfflineUpdateManager::applyDifferential(const UpdatePackage& basePkg, const UpdatePackage& patchPkg)
{
    QString message;

    const QString baseArchive = storedPayloadPath(basePkg);
    if (!QFile::exists(baseArchive)) {
        if (!copyPackagePayload(basePkg, &message)) {
            m_lastError = message.isEmpty()
                ? tr("Brak zainstalowanej wersji bazowej %1.").arg(basePkg.id)
                : message;
            return false;
        }
    }

    if (!copyPackagePayload(patchPkg, &message)) {
        m_lastError = message;
        return false;
    }

    const QString patchArchive = storedPayloadPath(patchPkg);
    if (!QFile::exists(patchArchive)) {
        m_lastError = tr("Nie można odnaleźć pliku łatki różnicowej dla %1.").arg(patchPkg.id);
        return false;
    }

    const QString stagingRoot = QDir(m_installDir).filePath(QStringLiteral("%1/%2_%3")
                                                                .arg(QString::fromLatin1(kStagingDirectory),
                                                                     patchPkg.id,
                                                                     patchPkg.version));
    const QString baseDir = QDir(stagingRoot).filePath(QStringLiteral("base"));
    const QString patchDir = QDir(stagingRoot).filePath(QStringLiteral("patch"));

    if (!ensureEmptyDirectory(baseDir, &message) || !ensureEmptyDirectory(patchDir, &message)) {
        m_lastError = message;
        return false;
    }

    if (!extractArchive(baseArchive, baseDir, &message)) {
        m_lastError = message;
        QDir(stagingRoot).removeRecursively();
        return false;
    }

    if (!extractArchive(patchArchive, patchDir, &message)) {
        m_lastError = message;
        QDir(stagingRoot).removeRecursively();
        return false;
    }

    if (!overlayDirectory(patchDir, baseDir, &message)) {
        m_lastError = message;
        QDir(stagingRoot).removeRecursively();
        return false;
    }

    const QVariant removalsValue = patchPkg.metadata.value(QStringLiteral("removals"));
    QStringList removals;
    if (removalsValue.canConvert<QVariantList>()) {
        const QVariantList list = removalsValue.toList();
        for (const QVariant& value : list)
            removals.append(value.toString());
    } else if (removalsValue.typeId() == QMetaType::QString) {
        removals.append(removalsValue.toString());
    }

    if (!removals.isEmpty() && !removePaths(removals, baseDir, &message)) {
        m_lastError = message;
        QDir(stagingRoot).removeRecursively();
        return false;
    }

    const QString targetArchive = storedArchivePath(patchPkg);
    QFile::remove(targetArchive);
    if (!createArchive(baseDir, targetArchive, &message)) {
        m_lastError = message;
        QDir(stagingRoot).removeRecursively();
        return false;
    }

    QDir(stagingRoot).removeRecursively();

    QVariantMap record;
    record.insert(QStringLiteral("id"), patchPkg.id);
    record.insert(QStringLiteral("version"), patchPkg.version);
    record.insert(QStringLiteral("baseId"), basePkg.id);
    record.insert(QStringLiteral("appliedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
    record.insert(QStringLiteral("metadata"), patchPkg.metadata);
    record.insert(QStringLiteral("artifact"), targetArchive);

    bool updated = false;
    for (int i = 0; i < m_installedUpdates.size(); ++i) {
        QVariantMap entry = m_installedUpdates.at(i).toMap();
        if (entry.value(QStringLiteral("id")).toString() == patchPkg.id) {
            m_installedUpdates[i] = record;
            updated = true;
            break;
        }
    }
    if (!updated)
        m_installedUpdates.append(record);

    persistState();
    Q_EMIT installedUpdatesChanged();
    return true;
}


QString OfflineUpdateManager::storedPayloadPath(const UpdatePackage& pkg) const
{
    QDir installDir(m_installDir);
    const QString extension = pkg.differential ? QStringLiteral("patch") : QStringLiteral("tar");
    return installDir.filePath(QStringLiteral("%1_%2.%3")
                                  .arg(pkg.id, pkg.version, extension));
}

QString OfflineUpdateManager::storedArchivePath(const UpdatePackage& pkg) const
{
    QDir installDir(m_installDir);
    return installDir.filePath(QStringLiteral("%1_%2.tar")
                                  .arg(pkg.id, pkg.version));
}

QString OfflineUpdateManager::packagePayloadPath(const UpdatePackage& pkg) const
{
    QString relative;
    if (pkg.differential) {
        relative = !pkg.diffFile.isEmpty() ? pkg.diffFile : QString::fromLatin1(kPatchFileName);
    } else {
        relative = !pkg.payloadFile.isEmpty() ? pkg.payloadFile : QString::fromLatin1(kPayloadFileName);
    }
    return QDir(pkg.path).filePath(relative);
}

bool OfflineUpdateManager::ensureEmptyDirectory(const QString& path, QString* errorMessage) const
{
    QDir dir(path);
    if (dir.exists() && !dir.removeRecursively()) {
        if (errorMessage)
            *errorMessage = tr("Nie można wyczyścić katalogu tymczasowego %1.").arg(path);
        return false;
    }
    if (!dir.mkpath(QStringLiteral("."))) {
        if (errorMessage)
            *errorMessage = tr("Nie można utworzyć katalogu %1.").arg(path);
        return false;
    }
    return true;
}

bool OfflineUpdateManager::extractArchive(const QString& archivePath, const QString& targetDir, QString* errorMessage) const
{
    if (!ensureEmptyDirectory(targetDir, errorMessage))
        return false;

    QStringList args;
    args << QStringLiteral("-xf") << normalizeForProcess(archivePath)
         << QStringLiteral("-C") << normalizeForProcess(targetDir);
    QString cliError;
    if (runArchiveCommand(args, &cliError))
        return true;

    QString fallbackError;
    if (extractArchiveWithLibArchive(archivePath, targetDir, &fallbackError))
        return true;

    if (errorMessage) {
        if (!cliError.isEmpty())
            *errorMessage = cliError;
        else if (!fallbackError.isEmpty())
            *errorMessage = fallbackError;
        else
            *errorMessage = tr("Nie można rozpakować archiwum %1.").arg(archivePath);
    }
    return false;
}

bool OfflineUpdateManager::createArchive(const QString& sourceDir, const QString& archivePath, QString* errorMessage) const
{
    QFile::remove(archivePath);
    QDir dir(sourceDir);
    if (!dir.exists()) {
        if (errorMessage)
            *errorMessage = tr("Katalog źródłowy %1 nie istnieje.").arg(sourceDir);
        return false;
    }

    QStringList args;
    args << QStringLiteral("-cf") << normalizeForProcess(archivePath)
         << QStringLiteral("-C") << normalizeForProcess(sourceDir)
         << QStringLiteral(".");
    QString cliError;
    if (!runArchiveCommand(args, &cliError)) {
        QString fallbackError;
        if (!createArchiveWithLibArchive(sourceDir, archivePath, &fallbackError)) {
            if (errorMessage) {
                if (!cliError.isEmpty())
                    *errorMessage = cliError;
                else if (!fallbackError.isEmpty())
                    *errorMessage = fallbackError;
                else
                    *errorMessage = tr("Nie można utworzyć archiwum %1.").arg(archivePath);
            }
            return false;
        }
    }
    return QFile::exists(archivePath);
}

bool OfflineUpdateManager::overlayDirectory(const QString& sourceDir, const QString& targetDir, QString* errorMessage) const
{
    QDir source(sourceDir);
    if (!source.exists())
        return true;

    if (!QDir(targetDir).exists() && !QDir().mkpath(targetDir)) {
        if (errorMessage)
            *errorMessage = tr("Nie można utworzyć katalogu docelowego %1.").arg(targetDir);
        return false;
    }

    QDirIterator it(sourceDir, QDir::NoDotAndDotDot | QDir::AllEntries, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString sourcePath = it.next();
        const QFileInfo info = it.fileInfo();
        const QString relativePath = source.relativeFilePath(sourcePath);
        const QString destinationPath = QDir(targetDir).filePath(relativePath);

        if (info.isDir()) {
            if (!QDir(destinationPath).exists() && !QDir().mkpath(destinationPath)) {
                if (errorMessage)
                    *errorMessage = tr("Nie można utworzyć katalogu %1.").arg(destinationPath);
                return false;
            }
            continue;
        }

        QDir destDir = QFileInfo(destinationPath).dir();
        if (!destDir.exists() && !destDir.mkpath(QStringLiteral("."))) {
            if (errorMessage)
                *errorMessage = tr("Nie można utworzyć katalogu %1.").arg(destDir.absolutePath());
            return false;
        }

        QFile::remove(destinationPath);
        if (!QFile::copy(sourcePath, destinationPath)) {
            if (errorMessage)
                *errorMessage = tr("Nie można skopiować pliku %1.").arg(sourcePath);
            return false;
        }
        QFile::setPermissions(destinationPath, info.permissions());
    }
    return true;
}

bool OfflineUpdateManager::removePaths(const QStringList& relativePaths, const QString& rootDir, QString* errorMessage) const
{
    QDir root(rootDir);
    for (const QString& relative : relativePaths) {
        QString normalized = QDir::cleanPath(relative);
        if (normalized.startsWith(QStringLiteral("..")))
            continue;
        const QString absolutePath = root.filePath(normalized);
        QFileInfo info(absolutePath);
        if (!info.exists())
            continue;
        if (info.isDir()) {
            QDir dir(absolutePath);
            if (!dir.removeRecursively()) {
                if (errorMessage)
                    *errorMessage = tr("Nie można usunąć katalogu %1.").arg(absolutePath);
                return false;
            }
        } else {
            if (!QFile::remove(absolutePath)) {
                if (errorMessage)
                    *errorMessage = tr("Nie można usunąć pliku %1.").arg(absolutePath);
                return false;
            }
        }
    }
    return true;
}

QStringList OfflineUpdateManager::installedIds() const
{
    QStringList ids;
    ids.reserve(m_installedUpdates.size());
    for (const QVariant& value : m_installedUpdates)
        ids.append(value.toMap().value(QStringLiteral("id")).toString());
    return ids;
}

OfflineUpdateManager::UpdatePackage OfflineUpdateManager::packageById(const QString& id) const
{
    QString error;
    const QList<UpdatePackage> packages = loadPackages(&error);
    for (const UpdatePackage& pkg : packages) {
        if (pkg.id == id)
            return pkg;
    }
    return UpdatePackage{};
}
