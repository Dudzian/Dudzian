#pragma once

#include <QAbstractListModel>
#include <QHash>
#include <QStringList>
#include <QVector>

#include <optional>

#include "app/UiModuleManager.hpp"

class UiModuleViewsModel : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(QString categoryFilter READ categoryFilter WRITE setCategoryFilter NOTIFY categoryFilterChanged)

public:
    enum Roles {
        IdRole = Qt::UserRole + 1,
        NameRole,
        SourceRole,
        CategoryRole,
        ModuleIdRole,
        MetadataRole,
    };

    explicit UiModuleViewsModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;

    QString categoryFilter() const { return m_categoryFilter; }
    void setCategoryFilter(const QString& category);

    void setModuleManager(UiModuleManager* manager);
    UiModuleManager* moduleManager() const { return m_manager; }

    Q_INVOKABLE QVariantMap viewAt(int index) const;
    Q_INVOKABLE QVariantMap findById(const QString& viewId) const;
    Q_INVOKABLE QStringList categories() const;

signals:
    void categoryFilterChanged();

private slots:
    void handleViewRegistered(const QString& moduleId, const QVariantMap& descriptor);
    void handleViewUnregistered(const QString& moduleId, const QString& viewId);
    void handleManagerDestroyed();

private:
    struct ViewEntry {
        QString moduleId;
        UiModuleManager::ViewDescriptor descriptor;
    };

    std::optional<ViewEntry> entryFromMap(const QString& moduleId, const QVariantMap& map) const;
    QVector<QString> buildOrderedIds() const;
    void rebuild();
    QVariantMap toVariantMap(const ViewEntry& entry) const;

    UiModuleManager* m_manager = nullptr;
    QMetaObject::Connection m_destroyConnection;
    QHash<QString, ViewEntry> m_registry;  // key = view id
    QVector<QString> m_orderedIds;
    QString m_categoryFilter;
};

