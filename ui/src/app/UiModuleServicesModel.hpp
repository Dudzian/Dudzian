#pragma once

#include <QAbstractListModel>
#include <QHash>
#include <QString>
#include <QVariant>
#include <QVector>

#include <optional>

#include "app/UiModuleManager.hpp"

class UiModuleServicesModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(QString searchFilter READ searchFilter WRITE setSearchFilter NOTIFY searchFilterChanged)

public:
    enum Roles {
        IdRole = Qt::UserRole + 1,
        NameRole,
        ModuleIdRole,
        SingletonRole,
        MetadataRole,
    };

    explicit UiModuleServicesModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;

    QString searchFilter() const { return m_searchFilter; }
    void setSearchFilter(const QString& query);

    void setModuleManager(UiModuleManager* manager);
    UiModuleManager* moduleManager() const { return m_manager; }

    Q_INVOKABLE QVariantMap serviceAt(int index) const;
    Q_INVOKABLE QVariantMap findById(const QString& serviceId) const;

signals:
    void searchFilterChanged();

private slots:
    void handleServiceRegistered(const QString& moduleId, const QVariantMap& descriptor);
    void handleServiceUnregistered(const QString& moduleId, const QString& serviceId);
    void handleManagerDestroyed();

private:
    struct ServiceEntry {
        QString moduleId;
        UiModuleManager::ServiceDescriptor descriptor;
    };

    std::optional<ServiceEntry> entryFromMap(const QString& moduleId, const QVariantMap& map) const;
    QVector<QString> buildOrderedIds() const;
    void rebuild();
    QVariantMap toVariantMap(const ServiceEntry& entry) const;
    bool matchesSearch(const ServiceEntry& entry, const QString& query) const;
    bool metadataContains(const QVariant& value, const QString& query) const;

    UiModuleManager* m_manager = nullptr;
    QMetaObject::Connection m_destroyConnection;
    QHash<QString, ServiceEntry> m_registry;  // key = service id
    QVector<QString> m_orderedIds;
    QString m_searchFilter;
};

