#include "OfflineUpdateManager.hpp"

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

QVariantMap jsonToVariant(const QJsonObject& object)
{
    QVariantMap map;
    for (auto it = object.begin(); it != object.end(); ++it)
        map.insert(it.key(), it.value().toVariant());
    return map;
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
    const QString manifestPath = QDir(pkg.path).filePath(QString::fromLatin1(kManifestFileName));
    if (pkg.signature.trimmed().isEmpty()) {
        if (message)
            *message = tr("Pakiet %1 nie zawiera podpisu.").arg(pkg.id);
        return false;
    }

    const QString payloadPath = QDir(pkg.path).filePath(pkg.differential ? QString::fromLatin1(kPatchFileName)
                                                                         : QString::fromLatin1(kPayloadFileName));
    const QString digest = computeFileSha256(payloadPath);
    if (digest.isEmpty()) {
        if (message)
            *message = tr("Nie można obliczyć skrótu pakietu %1.").arg(pkg.id);
        return false;
    }
    if (!pkg.signature.startsWith(digest.left(32))) {
        if (message)
            *message = tr("Podpis pakietu %1 nie zgadza się z manifestem.").arg(pkg.id);
        qCWarning(lcOfflineUpdates) << "Manifest" << manifestPath << "hash" << digest << "signature" << pkg.signature;
        return false;
    }
    Q_UNUSED(manifestPath);
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
        QFile file(m_tpmEvidencePath);
        if (file.open(QIODevice::ReadOnly)) {
            const QByteArray payload = file.readAll();
            file.close();
            const QString attestationHash = QString::fromUtf8(payload).trimmed();
            if (!attestationHash.startsWith(fingerprint.left(16))) {
                if (message)
                    *message = tr("Dowód TPM nie pasuje do fingerprintu urządzenia.");
                return false;
            }
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
    QDir dir(m_packagesDir);
    if (!dir.exists()) {
        if (errorMessage)
            *errorMessage = tr("Katalog aktualizacji %1 nie istnieje.").arg(m_packagesDir);
        return packages;
    }

    const QFileInfoList entries = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (const QFileInfo& info : entries) {
        const QString manifestPath = info.absoluteFilePath() + QLatin1Char('/') + QLatin1String(kManifestFileName);
        QFile manifest(manifestPath);
        if (!manifest.open(QIODevice::ReadOnly | QIODevice::Text))
            continue;
        const QByteArray payload = manifest.readAll();
        manifest.close();
        const QJsonDocument doc = QJsonDocument::fromJson(payload);
        if (!doc.isObject())
            continue;
        const QJsonObject root = doc.object();
        UpdatePackage pkg;
        pkg.path = info.absoluteFilePath();
        pkg.id = root.value(QStringLiteral("id")).toString(info.fileName());
        pkg.version = root.value(QStringLiteral("version")).toString();
        pkg.fingerprint = root.value(QStringLiteral("fingerprint")).toString();
        pkg.signature = root.value(QStringLiteral("signature")).toString();
        pkg.differential = root.value(QStringLiteral("differential")).toBool(false);
        pkg.baseId = root.value(QStringLiteral("baseId")).toString();
        pkg.metadata = jsonToVariant(root);
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
    const QString fileName = pkg.differential ? QString::fromLatin1(kPatchFileName)
                                              : QString::fromLatin1(kPayloadFileName);
    return QDir(pkg.path).filePath(fileName);
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
    return runArchiveCommand(args, errorMessage);
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
    if (!runArchiveCommand(args, errorMessage))
        return false;
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
