#include "app/UiModuleManager.hpp"

#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QLoggingCategory>
#include <QPluginLoader>
#include <QVariantList>
#include <QVariantMap>

#include <algorithm>

namespace {

bool hasLibrarySuffix(const QString& path)
{
#if defined(Q_OS_WIN)
    return path.endsWith(QStringLiteral(".dll"), Qt::CaseInsensitive);
#elif defined(Q_OS_MAC)
    return path.endsWith(QStringLiteral(".dylib"), Qt::CaseInsensitive) ||
           path.endsWith(QStringLiteral(".so"), Qt::CaseInsensitive);
#else
    return path.endsWith(QStringLiteral(".so"), Qt::CaseInsensitive);
#endif
}

} // namespace

UiModuleManager::UiModuleManager(QObject* parent)
    : QObject(parent)
{
}

QVariantMap UiModuleManager::LoadReport::toVariantMap() const
{
    QVariantMap map;
    map.insert(QStringLiteral("requestedPaths"), requestedPaths);
    map.insert(QStringLiteral("missingPaths"), missingPaths);
    map.insert(QStringLiteral("invalidEntries"), invalidEntries);
    map.insert(QStringLiteral("loadedPlugins"), loadedPlugins);
    map.insert(QStringLiteral("directoriesScanned"), directoriesScanned);
    map.insert(QStringLiteral("filesScanned"), filesScanned);
    map.insert(QStringLiteral("pluginsLoaded"), pluginsLoaded);
    map.insert(QStringLiteral("viewsRegistered"), viewsRegistered);
    map.insert(QStringLiteral("servicesRegistered"), servicesRegistered);

    QVariantList errorList;
    errorList.reserve(failedPlugins.size());
    for (const PluginLoadError& error : failedPlugins) {
        QVariantMap entry;
        entry.insert(QStringLiteral("path"), error.path);
        entry.insert(QStringLiteral("message"), error.message);
        errorList.append(entry);
    }
    map.insert(QStringLiteral("failedPlugins"), errorList);

    return map;
}

UiModuleManager::~UiModuleManager()
{
    unloadPlugins();
}

bool UiModuleManager::registerView(const QString& moduleId, const ViewDescriptor& descriptor)
{
    if (descriptor.id.isEmpty() || !descriptor.source.isValid()) {
        qWarning() << "UiModuleManager: pominięto widok z powodu niepoprawnych danych" << descriptor.id;
        return false;
    }
    if (m_views.contains(descriptor.id)) {
        qWarning() << "UiModuleManager: widok o identyfikatorze" << descriptor.id << "jest już zarejestrowany";
        return false;
    }

    ViewEntry entry{moduleId, descriptor};
    m_views.insert(descriptor.id, entry);
    emit viewRegistered(moduleId, serializeView(descriptor, moduleId));
    return true;
}

bool UiModuleManager::unregisterView(const QString& viewId)
{
    const auto it = m_views.find(viewId);
    if (it == m_views.end()) {
        return false;
    }
    emit viewUnregistered(it->moduleId, viewId);
    m_views.erase(it);
    return true;
}

bool UiModuleManager::registerService(const QString& moduleId, const ServiceDescriptor& descriptor)
{
    if (descriptor.id.isEmpty() || !descriptor.factory) {
        qWarning() << "UiModuleManager: pominięto serwis z powodu brakującej fabryki" << descriptor.id;
        return false;
    }
    if (m_services.contains(descriptor.id)) {
        qWarning() << "UiModuleManager: serwis o identyfikatorze" << descriptor.id << "jest już zarejestrowany";
        return false;
    }

    ServiceEntry entry{moduleId, descriptor, {}};
    m_services.insert(descriptor.id, entry);
    emit serviceRegistered(moduleId, serializeService(descriptor, moduleId));
    return true;
}

bool UiModuleManager::unregisterService(const QString& serviceId)
{
    const auto it = m_services.find(serviceId);
    if (it == m_services.end()) {
        return false;
    }
    emit serviceUnregistered(it->moduleId, serviceId);
    m_services.erase(it);
    return true;
}

QVariantList UiModuleManager::availableViews(const QString& category) const
{
    QVariantList result;
    result.reserve(m_views.size());
    for (auto it = m_views.constBegin(); it != m_views.constEnd(); ++it) {
        const ViewEntry& entry = it.value();
        if (!category.isEmpty() && entry.descriptor.category != category) {
            continue;
        }
        result.append(serializeView(entry.descriptor, entry.moduleId));
    }

    std::sort(result.begin(), result.end(), [](const QVariant& lhs, const QVariant& rhs) {
        const QVariantMap left = lhs.toMap();
        const QVariantMap right = rhs.toMap();
        return left.value(QStringLiteral("name")).toString() < right.value(QStringLiteral("name")).toString();
    });

    return result;
}

QVariantList UiModuleManager::availableServices() const
{
    QVariantList result;
    result.reserve(m_services.size());
    for (auto it = m_services.constBegin(); it != m_services.constEnd(); ++it) {
        const ServiceEntry& entry = it.value();
        result.append(serializeService(entry.descriptor, entry.moduleId));
    }

    std::sort(result.begin(), result.end(), [](const QVariant& lhs, const QVariant& rhs) {
        const QVariantMap left = lhs.toMap();
        const QVariantMap right = rhs.toMap();
        const QString leftName = left.value(QStringLiteral("name")).toString();
        const QString rightName = right.value(QStringLiteral("name")).toString();
        if (leftName == rightName)
            return left.value(QStringLiteral("id")).toString() < right.value(QStringLiteral("id")).toString();
        return leftName < rightName;
    });

    return result;
}

QVariantMap UiModuleManager::serviceDescriptor(const QString& serviceId) const
{
    const auto it = m_services.constFind(serviceId);
    if (it == m_services.constEnd())
        return {};
    return serializeService(it->descriptor, it->moduleId);
}

QObject* UiModuleManager::resolveService(const QString& serviceId) const
{
    auto it = m_services.find(serviceId);
    if (it == m_services.end()) {
        return nullptr;
    }
    return ensureServiceInstance(*it);
}

bool UiModuleManager::hasService(const QString& serviceId) const
{
    return m_services.contains(serviceId);
}

void UiModuleManager::addPluginPath(const QString& path)
{
    if (path.isEmpty()) {
        return;
    }
    if (!m_pluginPaths.contains(path)) {
        m_pluginPaths.append(path);
    }
}

void UiModuleManager::setPluginPaths(const QStringList& paths)
{
    m_pluginPaths = paths;
}

QStringList UiModuleManager::pluginPaths() const
{
    return m_pluginPaths;
}

bool UiModuleManager::loadPlugins(const QStringList& candidates)
{
    const QStringList targets = candidates.isEmpty() ? m_pluginPaths : candidates;
    bool success = true;

    LoadReport report;
    report.requestedPaths = targets;

    for (const QString& target : targets) {
        QFileInfo info(target);
        if (!info.exists()) {
            qWarning() << "UiModuleManager: pominięto ścieżkę pluginów" << target;
            success = false;
            report.missingPaths.append(target);
            continue;
        }

        if (info.isFile()) {
            report.filesScanned++;
            if (!isValidLibraryPath(info.absoluteFilePath())) {
                qWarning() << "UiModuleManager: pominięto nieobsługiwany plik" << info.absoluteFilePath();
                success = false;
                report.invalidEntries.append(info.absoluteFilePath());
                continue;
            }
            auto loader = std::make_unique<QPluginLoader>(info.absoluteFilePath());
            if (QObject* plugin = loader->instance()) {
                if (auto module = qobject_cast<UiModuleInterface*>(plugin)) {
                    module->registerComponents(*this);
                    m_pluginLoaders.push_back(std::move(loader));
                    report.loadedPlugins.append(info.absoluteFilePath());
                } else {
                    qWarning() << "UiModuleManager: plugin" << info.absoluteFilePath() << "nie implementuje UiModuleInterface";
                    loader->unload();
                    success = false;
                    report.failedPlugins.append({info.absoluteFilePath(),
                                                 QStringLiteral("Plugin nie implementuje UiModuleInterface")});
                }
            } else {
                qWarning() << "UiModuleManager: nie udało się załadować pluginu" << info.absoluteFilePath()
                           << loader->errorString();
                success = false;
                report.failedPlugins.append({info.absoluteFilePath(), loader->errorString()});
            }
            continue;
        }

        QDir dir(info.absoluteFilePath());
        report.directoriesScanned++;
        const QFileInfoList entries = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
        for (const QFileInfo& fileInfo : entries) {
            report.filesScanned++;
            if (!isValidLibraryPath(fileInfo.fileName())) {
                report.invalidEntries.append(fileInfo.absoluteFilePath());
                continue;
            }
            auto loader = std::make_unique<QPluginLoader>(fileInfo.absoluteFilePath());
            if (QObject* plugin = loader->instance()) {
                if (auto module = qobject_cast<UiModuleInterface*>(plugin)) {
                    module->registerComponents(*this);
                    m_pluginLoaders.push_back(std::move(loader));
                    report.loadedPlugins.append(fileInfo.absoluteFilePath());
                } else {
                    qWarning() << "UiModuleManager: plugin" << fileInfo.absoluteFilePath()
                               << "nie implementuje UiModuleInterface";
                    loader->unload();
                    success = false;
                    report.failedPlugins.append({fileInfo.absoluteFilePath(),
                                                 QStringLiteral("Plugin nie implementuje UiModuleInterface")});
                }
            } else {
                qWarning() << "UiModuleManager: nie udało się załadować pluginu" << fileInfo.absoluteFilePath()
                           << loader->errorString();
                success = false;
                report.failedPlugins.append({fileInfo.absoluteFilePath(), loader->errorString()});
            }
        }
    }

    report.pluginsLoaded = report.loadedPlugins.size();
    report.viewsRegistered = m_views.size();
    report.servicesRegistered = m_services.size();
    m_lastLoadReport = report;

    return success;
}

void UiModuleManager::unloadPlugins()
{
    for (auto it = m_services.begin(); it != m_services.end(); ++it) {
        if (QObject* instance = it->instance) {
            instance->deleteLater();
        }
        emit serviceUnregistered(it->moduleId, it.key());
    }
    m_services.clear();

    for (auto it = m_views.begin(); it != m_views.end(); ++it) {
        emit viewUnregistered(it->moduleId, it.key());
    }
    m_views.clear();

    for (auto& loader : m_pluginLoaders) {
        if (loader) {
            loader->unload();
        }
    }
    m_pluginLoaders.clear();
}

void UiModuleManager::registerModule(UiModuleInterface* module)
{
    if (!module) {
        return;
    }
    module->registerComponents(*this);
}

QVariantMap UiModuleManager::serializeView(const ViewDescriptor& descriptor, const QString& moduleId) const
{
    QVariantMap map;
    map.insert(QStringLiteral("id"), descriptor.id);
    map.insert(QStringLiteral("name"), descriptor.name);
    map.insert(QStringLiteral("source"), descriptor.source);
    map.insert(QStringLiteral("category"), descriptor.category);
    map.insert(QStringLiteral("moduleId"), moduleId);
    map.insert(QStringLiteral("metadata"), descriptor.metadata);
    return map;
}

QVariantMap UiModuleManager::serializeService(const ServiceDescriptor& descriptor, const QString& moduleId) const
{
    QVariantMap map;
    map.insert(QStringLiteral("id"), descriptor.id);
    map.insert(QStringLiteral("name"), descriptor.name);
    map.insert(QStringLiteral("moduleId"), moduleId);
    map.insert(QStringLiteral("singleton"), descriptor.singleton);
    map.insert(QStringLiteral("metadata"), descriptor.metadata);
    return map;
}

QObject* UiModuleManager::ensureServiceInstance(ServiceEntry& entry) const
{
    if (!entry.descriptor.factory) {
        return nullptr;
    }

    QObject* parent = const_cast<UiModuleManager*>(this);

    if (!entry.descriptor.singleton) {
        return entry.descriptor.factory(parent);
    }

    if (!entry.instance) {
        QObject* instance = entry.descriptor.factory(parent);
        if (!instance) {
            return nullptr;
        }
        entry.instance = instance;
    }
    return entry.instance;
}

bool UiModuleManager::isValidLibraryPath(const QString& path) const
{
    return hasLibrarySuffix(path);
}

QVariantMap UiModuleManager::lastLoadReport() const
{
    return m_lastLoadReport.toVariantMap();
}

void UiModuleManager::setLastLoadReportForTesting(const LoadReport& report)
{
    m_lastLoadReport = report;
}

