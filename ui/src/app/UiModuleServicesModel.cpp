#include "app/UiModuleServicesModel.hpp"

#include <QVariantList>
#include <QMetaType>

#include <algorithm>

namespace {

bool stringMatches(const QString& haystack, const QString& needle)
{
    return haystack.contains(needle, Qt::CaseInsensitive);
}

} // namespace

UiModuleServicesModel::UiModuleServicesModel(QObject* parent)
    : QAbstractListModel(parent)
{
}

int UiModuleServicesModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid())
        return 0;
    return m_orderedIds.size();
}

QVariant UiModuleServicesModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= m_orderedIds.size())
        return {};

    const QString& serviceId = m_orderedIds.at(index.row());
    const auto it = m_registry.constFind(serviceId);
    if (it == m_registry.constEnd())
        return {};

    const ServiceEntry& entry = it.value();
    switch (role) {
    case IdRole:
        return entry.descriptor.id;
    case NameRole:
        return entry.descriptor.name;
    case ModuleIdRole:
        return entry.moduleId;
    case SingletonRole:
        return entry.descriptor.singleton;
    case MetadataRole:
        return entry.descriptor.metadata;
    default:
        break;
    }
    return {};
}

QHash<int, QByteArray> UiModuleServicesModel::roleNames() const
{
    return {
        {IdRole, QByteArrayLiteral("id")},
        {NameRole, QByteArrayLiteral("name")},
        {ModuleIdRole, QByteArrayLiteral("moduleId")},
        {SingletonRole, QByteArrayLiteral("singleton")},
        {MetadataRole, QByteArrayLiteral("metadata")},
    };
}

void UiModuleServicesModel::setSearchFilter(const QString& query)
{
    if (m_searchFilter == query)
        return;
    m_searchFilter = query;
    emit searchFilterChanged();
    rebuild();
}

void UiModuleServicesModel::setModuleManager(UiModuleManager* manager)
{
    if (m_manager == manager)
        return;

    if (m_manager) {
        disconnect(m_manager, nullptr, this, nullptr);
        if (m_destroyConnection)
            disconnect(m_destroyConnection);
        m_destroyConnection = {};
    }

    beginResetModel();
    m_registry.clear();
    m_orderedIds.clear();
    m_manager = manager;

    if (m_manager) {
        const QVariantList descriptors = m_manager->availableServices();
        for (const QVariant& variant : descriptors) {
            const QVariantMap map = variant.toMap();
            const QString moduleId = map.value(QStringLiteral("moduleId")).toString();
            if (auto entry = entryFromMap(moduleId, map))
                m_registry.insert(entry->descriptor.id, *entry);
        }

        m_destroyConnection = connect(m_manager, &QObject::destroyed, this, &UiModuleServicesModel::handleManagerDestroyed);
        connect(m_manager,
                &UiModuleManager::serviceRegistered,
                this,
                &UiModuleServicesModel::handleServiceRegistered);
        connect(m_manager,
                &UiModuleManager::serviceUnregistered,
                this,
                &UiModuleServicesModel::handleServiceUnregistered);

        m_orderedIds = buildOrderedIds();
    }

    endResetModel();
}

QVariantMap UiModuleServicesModel::serviceAt(int index) const
{
    if (index < 0 || index >= m_orderedIds.size())
        return {};
    const QString& serviceId = m_orderedIds.at(index);
    const auto it = m_registry.constFind(serviceId);
    if (it == m_registry.constEnd())
        return {};
    return toVariantMap(it.value());
}

QVariantMap UiModuleServicesModel::findById(const QString& serviceId) const
{
    const auto it = m_registry.constFind(serviceId);
    if (it == m_registry.constEnd())
        return {};
    return toVariantMap(it.value());
}

void UiModuleServicesModel::handleServiceRegistered(const QString& moduleId, const QVariantMap& descriptor)
{
    if (auto entry = entryFromMap(moduleId, descriptor)) {
        m_registry.insert(entry->descriptor.id, *entry);
        rebuild();
    }
}

void UiModuleServicesModel::handleServiceUnregistered(const QString& moduleId, const QString& serviceId)
{
    Q_UNUSED(moduleId);
    if (!m_registry.contains(serviceId))
        return;

    m_registry.remove(serviceId);
    rebuild();
}

void UiModuleServicesModel::handleManagerDestroyed()
{
    beginResetModel();
    m_registry.clear();
    m_orderedIds.clear();
    m_manager = nullptr;
    endResetModel();
}

std::optional<UiModuleServicesModel::ServiceEntry> UiModuleServicesModel::entryFromMap(const QString& moduleId,
                                                                                       const QVariantMap& map) const
{
    if (!map.contains(QStringLiteral("id")))
        return std::nullopt;

    UiModuleManager::ServiceDescriptor descriptor;
    descriptor.id = map.value(QStringLiteral("id")).toString();
    descriptor.name = map.value(QStringLiteral("name")).toString();
    descriptor.metadata = map.value(QStringLiteral("metadata")).toMap();
    descriptor.singleton = map.value(QStringLiteral("singleton"), true).toBool();

    return ServiceEntry{moduleId, descriptor};
}

QVector<QString> UiModuleServicesModel::buildOrderedIds() const
{
    QVector<QString> ids;
    ids.reserve(m_registry.size());
    for (auto it = m_registry.constBegin(); it != m_registry.constEnd(); ++it) {
        const ServiceEntry& entry = it.value();
        if (!matchesSearch(entry, m_searchFilter))
            continue;
        ids.append(it.key());
    }

    std::sort(ids.begin(), ids.end(), [this](const QString& left, const QString& right) {
        const QString leftName = m_registry.value(left).descriptor.name;
        const QString rightName = m_registry.value(right).descriptor.name;
        if (leftName == rightName)
            return left < right;
        return leftName < rightName;
    });

    return ids;
}

void UiModuleServicesModel::rebuild()
{
    const QVector<QString> newOrder = buildOrderedIds();
    if (newOrder == m_orderedIds)
        return;

    beginResetModel();
    m_orderedIds = newOrder;
    endResetModel();
}

QVariantMap UiModuleServicesModel::toVariantMap(const ServiceEntry& entry) const
{
    QVariantMap map;
    map.insert(QStringLiteral("id"), entry.descriptor.id);
    map.insert(QStringLiteral("name"), entry.descriptor.name);
    map.insert(QStringLiteral("moduleId"), entry.moduleId);
    map.insert(QStringLiteral("singleton"), entry.descriptor.singleton);
    map.insert(QStringLiteral("metadata"), entry.descriptor.metadata);
    return map;
}

bool UiModuleServicesModel::matchesSearch(const ServiceEntry& entry, const QString& query) const
{
    if (query.isEmpty())
        return true;

    if (stringMatches(entry.descriptor.id, query))
        return true;
    if (stringMatches(entry.descriptor.name, query))
        return true;
    if (stringMatches(entry.moduleId, query))
        return true;
    if (metadataContains(entry.descriptor.metadata, query))
        return true;
    return false;
}

bool UiModuleServicesModel::metadataContains(const QVariant& value, const QString& query) const
{
    if (query.isEmpty())
        return false;

    if (value.typeId() == QMetaType::QString)
        return stringMatches(value.toString(), query);

    if (value.typeId() == QMetaType::QStringList) {
        const QStringList list = value.toStringList();
        for (const QString& item : list) {
            if (stringMatches(item, query))
                return true;
        }
        return false;
    }

    if (value.canConvert<QVariantMap>()) {
        const QVariantMap map = value.toMap();
        for (auto it = map.constBegin(); it != map.constEnd(); ++it) {
            if (stringMatches(it.key(), query))
                return true;
            if (metadataContains(it.value(), query))
                return true;
        }
        return false;
    }

    if (value.canConvert<QVariantList>()) {
        const QVariantList list = value.toList();
        for (const QVariant& item : list) {
            if (metadataContains(item, query))
                return true;
        }
        return false;
    }

    return stringMatches(value.toString(), query);
}

