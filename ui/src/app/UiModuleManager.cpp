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
    emit serviceRegistered(moduleId, descriptor.id);
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

    for (const QString& target : targets) {
        QFileInfo info(target);
        if (!info.exists()) {
            qWarning() << "UiModuleManager: pominięto ścieżkę pluginów" << target;
            success = false;
            continue;
        }

        if (info.isFile()) {
            if (!isValidLibraryPath(info.absoluteFilePath())) {
                qWarning() << "UiModuleManager: pominięto nieobsługiwany plik" << info.absoluteFilePath();
                success = false;
                continue;
            }
            auto loader = std::make_unique<QPluginLoader>(info.absoluteFilePath());
            if (QObject* plugin = loader->instance()) {
                if (auto module = qobject_cast<UiModuleInterface*>(plugin)) {
                    module->registerComponents(*this);
                    m_pluginLoaders.push_back(std::move(loader));
                } else {
                    qWarning() << "UiModuleManager: plugin" << info.absoluteFilePath() << "nie implementuje UiModuleInterface";
                    loader->unload();
                    success = false;
                }
            } else {
                qWarning() << "UiModuleManager: nie udało się załadować pluginu" << info.absoluteFilePath()
                           << loader->errorString();
                success = false;
            }
            continue;
        }

        QDir dir(info.absoluteFilePath());
        const QFileInfoList entries = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
        for (const QFileInfo& fileInfo : entries) {
            if (!isValidLibraryPath(fileInfo.fileName())) {
                continue;
            }
            auto loader = std::make_unique<QPluginLoader>(fileInfo.absoluteFilePath());
            if (QObject* plugin = loader->instance()) {
                if (auto module = qobject_cast<UiModuleInterface*>(plugin)) {
                    module->registerComponents(*this);
                    m_pluginLoaders.push_back(std::move(loader));
                } else {
                    qWarning() << "UiModuleManager: plugin" << fileInfo.absoluteFilePath()
                               << "nie implementuje UiModuleInterface";
                    loader->unload();
                    success = false;
                }
            } else {
                qWarning() << "UiModuleManager: nie udało się załadować pluginu" << fileInfo.absoluteFilePath()
                           << loader->errorString();
                success = false;
            }
        }
    }

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

