#pragma once

#include <QString>
#include <QtPlugin>

class UiModuleManager;

class UiModuleInterface {
public:
    virtual ~UiModuleInterface() = default;

    virtual QString moduleId() const = 0;
    virtual void registerComponents(UiModuleManager& manager) = 0;
};

#define UiModuleInterface_iid "com.dudzian.bot.ui.module/1.0"

Q_DECLARE_INTERFACE(UiModuleInterface, UiModuleInterface_iid)
