#include "app/UiModuleViewsModel.hpp"

#include <QVariantMap>
#include <QVariantList>

#include <algorithm>
#include <QSet>

UiModuleViewsModel::UiModuleViewsModel(QObject* parent)
    : QAbstractListModel(parent)
{
}

int UiModuleViewsModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid())
        return 0;
    return m_orderedIds.size();
}

QVariant UiModuleViewsModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= m_orderedIds.size())
        return {};

    const auto it = m_registry.constFind(m_orderedIds.at(index.row()));
    if (it == m_registry.constEnd())
        return {};

    const ViewEntry& entry = it.value();

    switch (role) {
    case IdRole:
        return entry.descriptor.id;
    case NameRole:
        return entry.descriptor.name;
    case SourceRole:
        return entry.descriptor.source;
    case CategoryRole:
        return entry.descriptor.category;
    case ModuleIdRole:
        return entry.moduleId;
    case MetadataRole:
        return entry.descriptor.metadata;
    default:
        break;
    }

    return {};
}

QHash<int, QByteArray> UiModuleViewsModel::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles.insert(IdRole, QByteArrayLiteral("id"));
    roles.insert(NameRole, QByteArrayLiteral("name"));
    roles.insert(SourceRole, QByteArrayLiteral("source"));
    roles.insert(CategoryRole, QByteArrayLiteral("category"));
    roles.insert(ModuleIdRole, QByteArrayLiteral("moduleId"));
    roles.insert(MetadataRole, QByteArrayLiteral("metadata"));
    return roles;
}

void UiModuleViewsModel::setCategoryFilter(const QString& category)
{
    if (m_categoryFilter == category)
        return;

    m_categoryFilter = category;
    emit categoryFilterChanged();
    rebuild();
}

void UiModuleViewsModel::setModuleManager(UiModuleManager* manager)
{
    if (m_manager == manager)
        return;

    if (m_manager) {
        disconnect(m_manager, nullptr, this, nullptr);
        if (m_destroyConnection)
            disconnect(m_destroyConnection);
    }

    beginResetModel();
    m_registry.clear();
    m_orderedIds.clear();

    m_manager = manager;

    if (m_manager) {
        m_destroyConnection = connect(m_manager, &QObject::destroyed, this, &UiModuleViewsModel::handleManagerDestroyed);
        connect(m_manager, &UiModuleManager::viewRegistered, this, &UiModuleViewsModel::handleViewRegistered);
        connect(m_manager, &UiModuleManager::viewUnregistered, this, &UiModuleViewsModel::handleViewUnregistered);

        const QVariantList views = m_manager->availableViews();
        for (const QVariant& variant : views) {
            const QVariantMap map = variant.toMap();
            const QString moduleId = map.value(QStringLiteral("moduleId")).toString();
            if (auto entry = entryFromMap(moduleId, map)) {
                m_registry.insert(entry->descriptor.id, *entry);
            }
        }

        m_orderedIds = buildOrderedIds();
    }

    endResetModel();
}

QVariantMap UiModuleViewsModel::viewAt(int index) const
{
    if (index < 0 || index >= m_orderedIds.size())
        return {};

    const auto it = m_registry.constFind(m_orderedIds.at(index));
    if (it == m_registry.constEnd())
        return {};

    return toVariantMap(it.value());
}

QVariantMap UiModuleViewsModel::findById(const QString& viewId) const
{
    const auto it = m_registry.constFind(viewId);
    if (it == m_registry.constEnd())
        return {};

    return toVariantMap(it.value());
}

QStringList UiModuleViewsModel::categories() const
{
    QSet<QString> unique;
    for (auto it = m_registry.cbegin(); it != m_registry.cend(); ++it) {
        const QString& category = it.value().descriptor.category;
        if (category.isEmpty())
            continue;
        unique.insert(category);
    }

    QStringList list = unique.values();
    std::sort(list.begin(), list.end(), [](const QString& lhs, const QString& rhs) {
        return lhs.localeAwareCompare(rhs) < 0;
    });
    return list;
}

void UiModuleViewsModel::handleViewRegistered(const QString& moduleId, const QVariantMap& descriptor)
{
    if (auto entry = entryFromMap(moduleId, descriptor)) {
        m_registry.insert(entry->descriptor.id, *entry);
        rebuild();
    }
}

void UiModuleViewsModel::handleViewUnregistered(const QString&, const QString& viewId)
{
    if (!m_registry.contains(viewId))
        return;

    m_registry.remove(viewId);
    rebuild();
}

void UiModuleViewsModel::handleManagerDestroyed()
{
    beginResetModel();
    m_registry.clear();
    m_orderedIds.clear();
    m_manager = nullptr;
    endResetModel();
}

std::optional<UiModuleViewsModel::ViewEntry> UiModuleViewsModel::entryFromMap(const QString& moduleId, const QVariantMap& map) const
{
    ViewEntry entry;
    entry.moduleId = moduleId;
    entry.descriptor.id = map.value(QStringLiteral("id")).toString();
    entry.descriptor.name = map.value(QStringLiteral("name")).toString();
    entry.descriptor.source = map.value(QStringLiteral("source")).toUrl();
    entry.descriptor.category = map.value(QStringLiteral("category")).toString();
    entry.descriptor.metadata = map.value(QStringLiteral("metadata")).toMap();

    if (entry.descriptor.id.isEmpty() || !entry.descriptor.source.isValid())
        return std::nullopt;

    return entry;
}

void UiModuleViewsModel::rebuild()
{
    beginResetModel();
    m_orderedIds = buildOrderedIds();
    endResetModel();
}

QVariantMap UiModuleViewsModel::toVariantMap(const ViewEntry& entry) const
{
    QVariantMap map;
    map.insert(QStringLiteral("id"), entry.descriptor.id);
    map.insert(QStringLiteral("name"), entry.descriptor.name);
    map.insert(QStringLiteral("source"), entry.descriptor.source);
    map.insert(QStringLiteral("category"), entry.descriptor.category);
    map.insert(QStringLiteral("moduleId"), entry.moduleId);
    map.insert(QStringLiteral("metadata"), entry.descriptor.metadata);
    return map;
}

QVector<QString> UiModuleViewsModel::buildOrderedIds() const
{
    QVector<QString> filtered;
    filtered.reserve(m_registry.size());
    for (auto it = m_registry.cbegin(); it != m_registry.cend(); ++it) {
        const ViewEntry& entry = it.value();
        if (!m_categoryFilter.isEmpty() && entry.descriptor.category != m_categoryFilter)
            continue;
        filtered.append(it.key());
    }

    std::sort(filtered.begin(), filtered.end(), [this](const QString& lhs, const QString& rhs) {
        const QString leftName = m_registry.value(lhs).descriptor.name;
        const QString rightName = m_registry.value(rhs).descriptor.name;
        if (leftName == rightName)
            return lhs < rhs;
        return leftName < rightName;
    });

    return filtered;
}

