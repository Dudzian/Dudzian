#pragma once

#include <QObject>
#include <QVariant>
#include <QHash>
#include <QSet>
#include <QStringList>

class StrategyWorkbenchController;

class UserProfileController : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString activeProfileId READ activeProfileId NOTIFY activeProfileChanged)
    Q_PROPERTY(QVariantMap activeProfile READ activeProfile NOTIFY activeProfileChanged)
    Q_PROPERTY(QVariantList profiles READ profiles NOTIFY profilesChanged)
    Q_PROPERTY(QStringList availableThemes READ availableThemes NOTIFY themesChanged)
    Q_PROPERTY(QVariantMap activeThemePalette READ activeThemePalette NOTIFY themePaletteChanged)
    Q_PROPERTY(QVariantMap activeWizardProgress READ activeWizardProgress NOTIFY wizardProgressChanged)
    Q_PROPERTY(QStringList catalogStrategyNames READ catalogStrategyNames NOTIFY catalogIntegrationChanged)
    Q_PROPERTY(QVariantList activeRecommendations READ activeRecommendations NOTIFY catalogIntegrationChanged)

public:
    explicit UserProfileController(QObject* parent = nullptr);
    ~UserProfileController() override;

    void setProfilesPath(const QString& path);
    void setStrategyCatalogController(StrategyWorkbenchController* controller);

    QString activeProfileId() const { return m_activeProfileId; }
    QVariantMap activeProfile() const;
    QVariantList profiles() const { return m_profiles; }
    QStringList availableThemes() const;
    QVariantMap activeThemePalette() const { return m_activeThemePalette; }
    QVariantMap activeWizardProgress() const { return m_activeWizardProgress; }
    QStringList catalogStrategyNames() const;
    QVariantList activeRecommendations() const;

    Q_INVOKABLE bool load();
    Q_INVOKABLE bool save() const;
    Q_INVOKABLE bool setActiveProfile(const QString& profileId);
    Q_INVOKABLE QVariantMap profileDetails(const QString& profileId) const;
    Q_INVOKABLE bool upsertProfile(const QVariantMap& profile);
    Q_INVOKABLE QString createProfile(const QString& displayName);
    Q_INVOKABLE QString duplicateProfile(const QString& sourceProfileId, const QString& displayName = QString());
    Q_INVOKABLE bool renameProfile(const QString& profileId, const QString& displayName);
    Q_INVOKABLE bool removeProfile(const QString& profileId);
    Q_INVOKABLE bool resetProfile(const QString& profileId);
    Q_INVOKABLE bool toggleFavoriteStrategy(const QString& profileId, const QString& strategyName);
    Q_INVOKABLE QStringList favoriteStrategies(const QString& profileId) const;
    Q_INVOKABLE QVariantList recommendedStrategies(const QString& profileId) const;
    Q_INVOKABLE bool applyTheme(const QString& profileId, const QString& themeId);
    Q_INVOKABLE QVariantMap themePalette(const QString& themeId) const;
    Q_INVOKABLE QVariantMap paletteOverrides(const QString& profileId) const;
    Q_INVOKABLE bool setPaletteOverride(const QString& profileId, const QString& role, const QString& colorValue);
    Q_INVOKABLE bool clearPaletteOverrides(const QString& profileId);
    Q_INVOKABLE QVariantMap wizardProgress(const QString& profileId) const;
    Q_INVOKABLE bool setWizardStepCompleted(const QString& profileId, const QString& stepId, bool completed = true);
    Q_INVOKABLE bool markWizardCompleted(const QString& profileId, bool completed = true);
    Q_INVOKABLE bool resetWizardProgress(const QString& profileId);
    Q_INVOKABLE void setCatalogSnapshot(const QVariantList& definitions);

signals:
    void profilesChanged();
    void activeProfileChanged();
    void themesChanged();
    void themePaletteChanged();
    void wizardProgressChanged();
    void catalogIntegrationChanged();

private:
    QVariantMap buildDefaultProfile(const QString& id) const;
    QString normalizeProfileId(const QString& candidate) const;
    QVariantMap sanitizedProfile(const QVariantMap& profile, const QString& skipId = QString()) const;
    bool writeProfiles(const QVariantMap& payload) const;
    QVariantMap readProfiles() const;
    bool ensureDefaultProfile();
    void rebuildThemePalette();
    void updateActiveRecommendations();
    void updateCatalogStrategyCache(const QVariantList& definitions);
    bool updateAvailableThemes();
    bool persistProfiles();
    bool applyActiveProfileTheme();
    bool upsertProfileInternal(const QVariantMap& profile, QString* assignedId);
    bool setActiveProfileInternal(const QString& profileId, bool emitSignals, bool* changed = nullptr);
    QVariantList collectSortedDefinitions(const QVariantList& definitions, const QStringList& exclude) const;
    QString ensureUniqueProfileId(const QString& baseId, const QString& skipId = QString()) const;
    bool profileIdExists(const QString& id, const QString& skipId = QString()) const;
    QVariantMap sanitizedPaletteOverrides(const QVariantMap& overrides) const;
    QVariantMap sanitizedWizardProgress(const QVariantMap& progress) const;
    QVariantMap defaultWizardProgress() const;
    bool isValidColorCode(const QString& color) const;
    QString normalizeColorCode(const QString& color) const;
    void updateActiveWizardProgress();

    QString m_profilesPath;
    QVariantList m_profiles;
    QString m_activeProfileId;
    QVariantMap m_activeThemePalette;
    QVariantMap m_activeWizardProgress;
    QStringList m_paletteRoles;

    StrategyWorkbenchController* m_catalogController = nullptr;
    QSet<QString> m_catalogStrategyIds;
    QVariantList m_catalogDefinitions;

    QHash<QString, QVariantMap> m_themePalettes;
};
