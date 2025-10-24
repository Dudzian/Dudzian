#pragma once

#include <QObject>
#include <QHash>
#include <QPointer>
#include <QStringList>
#include <QUrl>
#include <QVariantMap>
#include <QVector>

#include <functional>
#include <memory>

#include "app/UiModuleInterface.hpp"

class QPluginLoader;

class UiModuleManager : public QObject {
    Q_OBJECT

public:
    struct ViewDescriptor {
        QString id;
        QString name;
        QUrl source;
        QString category;
        QVariantMap metadata;
    };

    struct ServiceDescriptor {
        QString id;
        QString name;
        QVariantMap metadata;
        std::function<QObject*(QObject*)> factory;
        bool singleton = true;
    };

    explicit UiModuleManager(QObject* parent = nullptr);
    ~UiModuleManager() override;

    bool registerView(const QString& moduleId, const ViewDescriptor& descriptor);
    bool unregisterView(const QString& viewId);

    bool registerService(const QString& moduleId, const ServiceDescriptor& descriptor);
    bool unregisterService(const QString& serviceId);

    Q_INVOKABLE QVariantList availableViews(const QString& category = QString()) const;
    Q_INVOKABLE QObject* resolveService(const QString& serviceId) const;

    bool hasService(const QString& serviceId) const;

    void addPluginPath(const QString& path);
    void setPluginPaths(const QStringList& paths);
    QStringList pluginPaths() const;

    virtual bool loadPlugins(const QStringList& candidates = {});
    void unloadPlugins();

    void registerModule(UiModuleInterface* module);

signals:
    void viewRegistered(const QString& moduleId, const QVariantMap& descriptor);
    void viewUnregistered(const QString& moduleId, const QString& viewId);
    void serviceRegistered(const QString& moduleId, const QString& serviceId);
    void serviceUnregistered(const QString& moduleId, const QString& serviceId);

private:
    struct ViewEntry {
        QString moduleId;
        ViewDescriptor descriptor;
    };

    struct ServiceEntry {
        QString moduleId;
        ServiceDescriptor descriptor;
        mutable QPointer<QObject> instance;
    };

    QVariantMap serializeView(const ViewDescriptor& descriptor, const QString& moduleId) const;
    QObject* ensureServiceInstance(ServiceEntry& entry) const;
    bool isValidLibraryPath(const QString& path) const;

    QHash<QString, ViewEntry> m_views;
    QHash<QString, ServiceEntry> m_services;
    QStringList m_pluginPaths;
    QVector<std::unique_ptr<QPluginLoader>> m_pluginLoaders;
};

