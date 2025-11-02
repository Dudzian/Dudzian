#include "UserProfileController.hpp"

#include "StrategyWorkbenchController.hpp"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QList>
#include <QLoggingCategory>
#include <QRegularExpression>
#include <QtGlobal>

#include <algorithm>

Q_LOGGING_CATEGORY(lcUserProfiles, "bot.shell.ui.user_profiles")

namespace {
QVariantMap makePalette(const QString& backgroundPrimary,
                        const QString& backgroundOverlay,
                        const QString& surfaceStrong,
                        const QString& surfaceMuted,
                        const QString& surfaceSubtle,
                        const QString& textPrimary,
                        const QString& textSecondary,
                        const QString& textTertiary,
                        const QString& accent,
                        const QString& accentMuted,
                        const QString& positive,
                        const QString& negative,
                        const QString& warning)
{
    QVariantMap palette;
    palette.insert(QStringLiteral("backgroundPrimary"), backgroundPrimary);
    palette.insert(QStringLiteral("backgroundOverlay"), backgroundOverlay);
    palette.insert(QStringLiteral("surfaceStrong"), surfaceStrong);
    palette.insert(QStringLiteral("surfaceMuted"), surfaceMuted);
    palette.insert(QStringLiteral("surfaceSubtle"), surfaceSubtle);
    palette.insert(QStringLiteral("textPrimary"), textPrimary);
    palette.insert(QStringLiteral("textSecondary"), textSecondary);
    palette.insert(QStringLiteral("textTertiary"), textTertiary);
    palette.insert(QStringLiteral("accent"), accent);
    palette.insert(QStringLiteral("accentMuted"), accentMuted);
    palette.insert(QStringLiteral("positive"), positive);
    palette.insert(QStringLiteral("negative"), negative);
    palette.insert(QStringLiteral("warning"), warning);
    return palette;
}

QString timestampUtc()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
}
} // namespace

UserProfileController::UserProfileController(QObject* parent)
    : QObject(parent)
{
    m_themePalettes.insert(QStringLiteral("midnight"),
                           makePalette(QStringLiteral("#0E1320"),
                                       QStringLiteral("#161C2A"),
                                       QStringLiteral("#1F2536"),
                                       QStringLiteral("#242B3D"),
                                       QStringLiteral("#2C3448"),
                                       QStringLiteral("#F5F7FA"),
                                       QStringLiteral("#A4ACC4"),
                                       QStringLiteral("#7C86A4"),
                                       QStringLiteral("#4FA3FF"),
                                       QStringLiteral("#3577D4"),
                                       QStringLiteral("#3FD0A4"),
                                       QStringLiteral("#FF6B6B"),
                                       QStringLiteral("#F8C572")));
    m_themePalettes.insert(QStringLiteral("aurora"),
                           makePalette(QStringLiteral("#101926"),
                                       QStringLiteral("#1A2234"),
                                       QStringLiteral("#233044"),
                                       QStringLiteral("#2D3B52"),
                                       QStringLiteral("#354765"),
                                       QStringLiteral("#F0F7FF"),
                                       QStringLiteral("#B8C4E0"),
                                       QStringLiteral("#8DA1C7"),
                                       QStringLiteral("#8C7BFF"),
                                       QStringLiteral("#6A5DD8"),
                                       QStringLiteral("#4FDDB5"),
                                       QStringLiteral("#FF7F8C"),
                                       QStringLiteral("#FCD77F")));
    m_themePalettes.insert(QStringLiteral("solarized"),
                           makePalette(QStringLiteral("#002B36"),
                                       QStringLiteral("#073642"),
                                       QStringLiteral("#0B3A46"),
                                       QStringLiteral("#0F4552"),
                                       QStringLiteral("#13515F"),
                                       QStringLiteral("#FDF6E3"),
                                       QStringLiteral("#EEE8D5"),
                                       QStringLiteral("#C7BFA6"),
                                       QStringLiteral("#B58900"),
                                       QStringLiteral("#9C7A00"),
                                       QStringLiteral("#859900"),
                                       QStringLiteral("#DC322F"),
                                       QStringLiteral("#CB4B16")));

    m_paletteRoles = m_themePalettes.value(QStringLiteral("midnight")).keys();
    std::sort(m_paletteRoles.begin(), m_paletteRoles.end());

    updateAvailableThemes();
    m_activeThemePalette = m_themePalettes.value(QStringLiteral("midnight"));
    m_activeWizardProgress = defaultWizardProgress();

    Q_EMIT themesChanged();
    Q_EMIT themePaletteChanged();
    Q_EMIT wizardProgressChanged();
}

UserProfileController::~UserProfileController() = default;

void UserProfileController::setProfilesPath(const QString& path)
{
    const QString trimmed = path.trimmed();
    if (m_profilesPath == trimmed)
        return;
    m_profilesPath = trimmed;
}

void UserProfileController::setStrategyCatalogController(StrategyWorkbenchController* controller)
{
    if (m_catalogController == controller)
        return;

    if (m_catalogController)
        disconnect(m_catalogController, nullptr, this, nullptr);

    m_catalogController = controller;

    if (m_catalogController) {
        connect(m_catalogController,
                &StrategyWorkbenchController::catalogChanged,
                this,
                [this]() {
                    setCatalogSnapshot(m_catalogController->catalogDefinitions());
                });
        setCatalogSnapshot(m_catalogController->catalogDefinitions());
    }
}

QVariantMap UserProfileController::activeProfile() const
{
    for (const QVariant& entry : m_profiles) {
        const QVariantMap map = entry.toMap();
        if (map.value(QStringLiteral("id")).toString() == m_activeProfileId)
            return map;
    }
    return {};
}

QStringList UserProfileController::availableThemes() const
{
    QStringList themes = m_themePalettes.keys();
    std::sort(themes.begin(), themes.end());
    return themes;
}

QStringList UserProfileController::catalogStrategyNames() const
{
    QStringList names = m_catalogStrategyIds.values();
    std::sort(names.begin(), names.end());
    return names;
}

QVariantList UserProfileController::activeRecommendations() const
{
    return recommendedStrategies(m_activeProfileId);
}

bool UserProfileController::load()
{
    const QVariantMap payload = readProfiles();
    m_profiles.clear();
    m_activeProfileId.clear();

    bool needsPersist = false;

    const QVariant profilesVariant = payload.value(QStringLiteral("profiles"));
    if (profilesVariant.canConvert<QVariantList>()) {
        const QVariantList list = profilesVariant.toList();
        for (const QVariant& entry : list) {
            const QVariantMap original = entry.toMap();
            const QVariantMap sanitized = sanitizedProfile(original);
            if (sanitized.isEmpty()) {
                if (!original.isEmpty())
                    needsPersist = true;
                continue;
            }
            if (sanitized != original)
                needsPersist = true;
            m_profiles.append(sanitized);
        }
    }

    if (ensureDefaultProfile())
        needsPersist = true;

    if (updateAvailableThemes())
        needsPersist = true;

    QString requestedActive = payload.value(QStringLiteral("active"), QStringLiteral("default")).toString();
    if (requestedActive.trimmed().isEmpty())
        requestedActive = QStringLiteral("default");

    bool activeChanged = false;
    if (!setActiveProfileInternal(requestedActive, /*emitSignals*/ false, &activeChanged)) {
        if (!m_profiles.isEmpty()) {
            const QVariantMap first = m_profiles.first().toMap();
            const QString fallbackId = first.value(QStringLiteral("id")).toString();
            if (setActiveProfileInternal(fallbackId, false, &activeChanged))
                needsPersist = true;
        }
    } else if (activeChanged) {
        needsPersist = true;
    }

    if (needsPersist && !persistProfiles())
        return false;

    updateActiveWizardProgress();

    Q_EMIT profilesChanged();
    return true;
}

bool UserProfileController::save() const
{
    return persistProfiles();
}

bool UserProfileController::setActiveProfile(const QString& profileId)
{
    bool changed = false;
    if (!setActiveProfileInternal(profileId, true, &changed))
        return false;
    if (!changed)
        return true;
    return persistProfiles();
}

QVariantMap UserProfileController::profileDetails(const QString& profileId) const
{
    const QString normalized = normalizeProfileId(profileId);
    for (const QVariant& entry : m_profiles) {
        const QVariantMap map = entry.toMap();
        if (map.value(QStringLiteral("id")).toString() == normalized)
            return map;
    }
    return {};
}

bool UserProfileController::upsertProfile(const QVariantMap& profile)
{
    return upsertProfileInternal(profile, nullptr);
}

QString UserProfileController::createProfile(const QString& displayName)
{
    QVariantMap profile;
    profile.insert(QStringLiteral("displayName"), displayName);

    QString assignedId;
    if (!upsertProfileInternal(profile, &assignedId))
        return {};

    if (assignedId.isEmpty())
        return {};

    bool changed = false;
    if (setActiveProfileInternal(assignedId, true, &changed) && changed)
        persistProfiles();

    return assignedId;
}

QString UserProfileController::duplicateProfile(const QString& sourceProfileId, const QString& displayName)
{
    const QString normalizedSource = normalizeProfileId(sourceProfileId.isEmpty() ? m_activeProfileId : sourceProfileId);
    if (normalizedSource.isEmpty())
        return {};

    QVariantMap source = profileDetails(normalizedSource);
    if (source.isEmpty())
        return {};

    const QString trimmedName = displayName.trimmed();
    QString finalName = trimmedName;
    if (finalName.isEmpty()) {
        const QString originalName = source.value(QStringLiteral("displayName")).toString();
        if (!originalName.trimmed().isEmpty())
            finalName = tr("%1 (kopia)").arg(originalName.trimmed());
        else
            finalName = tr("Profil %1 (kopia)").arg(normalizedSource);
    }

    source.remove(QStringLiteral("id"));
    source.remove(QStringLiteral("createdAt"));
    source.remove(QStringLiteral("updatedAt"));
    source.insert(QStringLiteral("displayName"), finalName);

    QString assignedId;
    if (!upsertProfileInternal(source, &assignedId))
        return {};

    if (assignedId.isEmpty())
        return {};

    bool changed = false;
    if (setActiveProfileInternal(assignedId, true, &changed) && changed)
        persistProfiles();

    return assignedId;
}

bool UserProfileController::renameProfile(const QString& profileId, const QString& displayName)
{
    const QString normalizedId = normalizeProfileId(profileId);
    if (normalizedId.isEmpty())
        return false;

    QVariantMap existing = profileDetails(normalizedId);
    if (existing.isEmpty())
        return false;

    existing.insert(QStringLiteral("displayName"), displayName);

    QString assignedId;
    if (!upsertProfileInternal(existing, &assignedId))
        return false;

    if (assignedId != normalizedId) {
        bool changed = false;
        if (setActiveProfileInternal(assignedId, true, &changed) && changed)
            persistProfiles();
    }

    return true;
}

bool UserProfileController::removeProfile(const QString& profileId)
{
    if (m_profiles.size() <= 1)
        return false;
    const QString normalized = normalizeProfileId(profileId);
    bool removed = false;
    for (int i = 0; i < m_profiles.size(); ++i) {
        if (m_profiles.at(i).toMap().value(QStringLiteral("id")).toString() == normalized) {
            m_profiles.removeAt(i);
            removed = true;
            break;
        }
    }
    if (!removed)
        return false;

    if (m_activeProfileId == normalized) {
        const QVariantMap first = m_profiles.first().toMap();
        m_activeProfileId = first.value(QStringLiteral("id")).toString();
        Q_EMIT activeProfileChanged();
    }

    updateAvailableThemes();
    applyActiveProfileTheme();
    updateActiveRecommendations();
    updateActiveWizardProgress();

    Q_EMIT profilesChanged();
    return persistProfiles();
}

bool UserProfileController::resetProfile(const QString& profileId)
{
    const QString normalizedProfile = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    if (normalizedProfile.isEmpty())
        return false;

    const QString defaultTheme = QStringLiteral("midnight");
    for (int i = 0; i < m_profiles.size(); ++i) {
        QVariantMap map = m_profiles.at(i).toMap();
        if (map.value(QStringLiteral("id")).toString() != normalizedProfile)
            continue;

        bool changed = false;
        if (!map.value(QStringLiteral("favorites")).toStringList().isEmpty()) {
            map.insert(QStringLiteral("favorites"), QStringList());
            changed = true;
        }

        if (map.value(QStringLiteral("theme")).toString() != defaultTheme) {
            map.insert(QStringLiteral("theme"), defaultTheme);
            changed = true;
        }

        if (!map.value(QStringLiteral("paletteOverrides")).toMap().isEmpty()) {
            map.remove(QStringLiteral("paletteOverrides"));
            changed = true;
        }

        const QVariantMap progress = sanitizedWizardProgress(map.value(QStringLiteral("setupProgress")).toMap());
        const QVariantMap defaults = defaultWizardProgress();
        if (progress != defaults) {
            map.insert(QStringLiteral("setupProgress"), defaults);
            changed = true;
        }

        if (!changed)
            return true;

        map.insert(QStringLiteral("updatedAt"), timestampUtc());
        m_profiles[i] = map;

        if (normalizedProfile == m_activeProfileId) {
            applyActiveProfileTheme();
            updateActiveRecommendations();
            updateActiveWizardProgress();
            Q_EMIT activeProfileChanged();
        }

        Q_EMIT profilesChanged();
        Q_EMIT themesChanged();
        if (!persistProfiles())
            qCWarning(lcUserProfiles) << "Failed to persist profile reset";
        return true;
    }

    return false;
}

bool UserProfileController::upsertProfileInternal(const QVariantMap& profile, QString* assignedId)
{
    QString skipId;
    const QString requestedId = normalizeProfileId(profile.value(QStringLiteral("id")).toString());
    if (!requestedId.isEmpty() && profileIdExists(requestedId))
        skipId = requestedId;

    QVariantMap sanitized = sanitizedProfile(profile, skipId);
    if (sanitized.isEmpty())
        return false;

    const QString id = sanitized.value(QStringLiteral("id")).toString();
    if (assignedId)
        *assignedId = id;

    const bool activeAffected = (id == m_activeProfileId);
    bool replaced = false;
    for (int i = 0; i < m_profiles.size(); ++i) {
        QVariantMap current = m_profiles.at(i).toMap();
        if (current.value(QStringLiteral("id")).toString() != id)
            continue;

        sanitized.insert(QStringLiteral("createdAt"), current.value(QStringLiteral("createdAt"), timestampUtc()));
        sanitized.insert(QStringLiteral("updatedAt"), timestampUtc());
        m_profiles[i] = sanitized;
        replaced = true;
        break;
    }

    if (!replaced) {
        const QString now = timestampUtc();
        sanitized.insert(QStringLiteral("createdAt"), sanitized.value(QStringLiteral("createdAt"), now));
        sanitized.insert(QStringLiteral("updatedAt"), now);
        m_profiles.append(sanitized);
    }

    if (m_activeProfileId.isEmpty())
        m_activeProfileId = id;

    updateAvailableThemes();
    applyActiveProfileTheme();
    updateActiveRecommendations();
    updateActiveWizardProgress();

    if (activeAffected)
        Q_EMIT activeProfileChanged();

    Q_EMIT profilesChanged();
    if (!persistProfiles())
        qCWarning(lcUserProfiles) << "Failed to persist profile changes";

    return true;
}

bool UserProfileController::toggleFavoriteStrategy(const QString& profileId, const QString& strategyName)
{
    const QString normalizedProfile = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    const QString trimmedStrategy = strategyName.trimmed();
    if (normalizedProfile.isEmpty() || trimmedStrategy.isEmpty())
        return false;
    if (!m_catalogStrategyIds.isEmpty() && !m_catalogStrategyIds.contains(trimmedStrategy))
        return false;

    for (int i = 0; i < m_profiles.size(); ++i) {
        QVariantMap map = m_profiles.at(i).toMap();
        if (map.value(QStringLiteral("id")).toString() != normalizedProfile)
            continue;

        QStringList favorites = map.value(QStringLiteral("favorites")).toStringList();
        const int index = favorites.indexOf(trimmedStrategy);
        if (index == -1)
            favorites.append(trimmedStrategy);
        else
            favorites.removeAt(index);
        map.insert(QStringLiteral("favorites"), favorites);
        m_profiles[i] = map;

        if (normalizedProfile == m_activeProfileId)
            Q_EMIT activeProfileChanged();

        Q_EMIT profilesChanged();
        updateActiveRecommendations();
        return persistProfiles();
    }

    return false;
}

QStringList UserProfileController::favoriteStrategies(const QString& profileId) const
{
    const QString normalized = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    for (const QVariant& entry : m_profiles) {
        const QVariantMap map = entry.toMap();
        if (map.value(QStringLiteral("id")).toString() == normalized)
            return map.value(QStringLiteral("favorites")).toStringList();
    }
    return {};
}

QVariantList UserProfileController::recommendedStrategies(const QString& profileId) const
{
    const QString normalized = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    const QStringList favorites = favoriteStrategies(normalized);
    return collectSortedDefinitions(m_catalogDefinitions, favorites);
}

bool UserProfileController::applyTheme(const QString& profileId, const QString& themeId)
{
    if (!m_themePalettes.contains(themeId))
        return false;

    const QString normalizedProfile = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    bool updated = false;
    for (int i = 0; i < m_profiles.size(); ++i) {
        QVariantMap map = m_profiles.at(i).toMap();
        if (map.value(QStringLiteral("id")).toString() != normalizedProfile)
            continue;
        if (map.value(QStringLiteral("theme")).toString() == themeId)
            return true;
        map.insert(QStringLiteral("theme"), themeId);
        map.insert(QStringLiteral("updatedAt"), timestampUtc());
        m_profiles[i] = map;
        updated = true;
        break;
    }

    if (!updated)
        return false;

    if (normalizedProfile == m_activeProfileId) {
        applyActiveProfileTheme();
        Q_EMIT activeProfileChanged();
    }

    Q_EMIT profilesChanged();
    Q_EMIT themesChanged();
    if (!persistProfiles())
        qCWarning(lcUserProfiles) << "Failed to persist theme change";
    return true;
}

QVariantMap UserProfileController::themePalette(const QString& themeId) const
{
    return m_themePalettes.value(themeId);
}

QVariantMap UserProfileController::paletteOverrides(const QString& profileId) const
{
    const QString normalized = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    for (const QVariant& entry : m_profiles) {
        const QVariantMap map = entry.toMap();
        if (map.value(QStringLiteral("id")).toString() == normalized)
            return sanitizedPaletteOverrides(map.value(QStringLiteral("paletteOverrides")).toMap());
    }
    return {};
}

bool UserProfileController::setPaletteOverride(const QString& profileId,
                                               const QString& role,
                                               const QString& colorValue)
{
    const QString normalizedProfile = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    const QString trimmedRole = role.trimmed();
    if (normalizedProfile.isEmpty() || !m_paletteRoles.contains(trimmedRole))
        return false;

    const QString trimmedColor = colorValue.trimmed();
    const bool remove = trimmedColor.isEmpty();
    QString normalizedColor = normalizeColorCode(trimmedColor);
    if (!remove && (normalizedColor.isEmpty() || !isValidColorCode(normalizedColor)))
        return false;

    for (int i = 0; i < m_profiles.size(); ++i) {
        QVariantMap map = m_profiles.at(i).toMap();
        if (map.value(QStringLiteral("id")).toString() != normalizedProfile)
            continue;

        QVariantMap overrides = map.value(QStringLiteral("paletteOverrides")).toMap();

        if (remove) {
            if (!overrides.contains(trimmedRole))
                return true;
            overrides.remove(trimmedRole);
        } else {
            const QString previous = normalizeColorCode(overrides.value(trimmedRole).toString());
            if (!previous.isEmpty() && previous.compare(normalizedColor, Qt::CaseInsensitive) == 0)
                return true;
            overrides.insert(trimmedRole, normalizedColor);
        }

        if (overrides.isEmpty())
            map.remove(QStringLiteral("paletteOverrides"));
        else
            map.insert(QStringLiteral("paletteOverrides"), overrides);

        map.insert(QStringLiteral("updatedAt"), timestampUtc());
        m_profiles[i] = map;

        if (normalizedProfile == m_activeProfileId)
            applyActiveProfileTheme();

        Q_EMIT profilesChanged();
        if (normalizedProfile == m_activeProfileId)
            Q_EMIT activeProfileChanged();
        const bool persisted = persistProfiles();
        if (!persisted)
            qCWarning(lcUserProfiles) << "Failed to persist palette override";
        return persisted;
    }

    return false;
}

bool UserProfileController::clearPaletteOverrides(const QString& profileId)
{
    const QString normalizedProfile = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    if (normalizedProfile.isEmpty())
        return false;

    for (int i = 0; i < m_profiles.size(); ++i) {
        QVariantMap map = m_profiles.at(i).toMap();
        if (map.value(QStringLiteral("id")).toString() != normalizedProfile)
            continue;

        if (!map.contains(QStringLiteral("paletteOverrides")) || map.value(QStringLiteral("paletteOverrides")).toMap().isEmpty())
            return true;

        map.remove(QStringLiteral("paletteOverrides"));
        map.insert(QStringLiteral("updatedAt"), timestampUtc());
        m_profiles[i] = map;

        if (normalizedProfile == m_activeProfileId)
            applyActiveProfileTheme();

        Q_EMIT profilesChanged();
        if (normalizedProfile == m_activeProfileId)
            Q_EMIT activeProfileChanged();
        const bool persisted = persistProfiles();
        if (!persisted)
            qCWarning(lcUserProfiles) << "Failed to persist palette reset";
        return persisted;
    }

    return false;
}

QVariantMap UserProfileController::wizardProgress(const QString& profileId) const
{
    const QString normalized = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    for (const QVariant& entry : m_profiles) {
        const QVariantMap map = entry.toMap();
        if (map.value(QStringLiteral("id")).toString() != normalized)
            continue;
        return sanitizedWizardProgress(map.value(QStringLiteral("setupProgress")).toMap());
    }
    return defaultWizardProgress();
}

bool UserProfileController::setWizardStepCompleted(const QString& profileId,
                                                   const QString& stepId,
                                                   bool completed)
{
    const QString normalizedProfile = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    const QString trimmedStep = stepId.trimmed();
    if (normalizedProfile.isEmpty() || trimmedStep.isEmpty())
        return false;

    for (int i = 0; i < m_profiles.size(); ++i) {
        QVariantMap map = m_profiles.at(i).toMap();
        if (map.value(QStringLiteral("id")).toString() != normalizedProfile)
            continue;

        QVariantMap progress = sanitizedWizardProgress(map.value(QStringLiteral("setupProgress")).toMap());
        QStringList steps = progress.value(QStringLiteral("completedSteps")).toStringList();
        const int index = steps.indexOf(trimmedStep);
        bool changed = false;

        if (completed) {
            if (index == -1) {
                steps.append(trimmedStep);
                changed = true;
            }
            const QString existingLast = progress.value(QStringLiteral("lastStep")).toString();
            if (existingLast != trimmedStep) {
                progress.insert(QStringLiteral("lastStep"), trimmedStep);
                changed = true;
            }
        } else {
            if (index != -1) {
                steps.removeAt(index);
                changed = true;
            }
            if (progress.value(QStringLiteral("lastStep")).toString() == trimmedStep) {
                progress.remove(QStringLiteral("lastStep"));
                changed = true;
            }
        }

        if (!changed)
            return true;

        bool completedState = progress.value(QStringLiteral("completed")).toBool();
        if (!completed && steps.isEmpty())
            completedState = false;
        progress.insert(QStringLiteral("completedSteps"), steps);
        progress.insert(QStringLiteral("completed"), completedState);
        progress.insert(QStringLiteral("updatedAt"), timestampUtc());
        map.insert(QStringLiteral("setupProgress"), progress);
        map.insert(QStringLiteral("updatedAt"), timestampUtc());
        m_profiles[i] = map;

        if (normalizedProfile == m_activeProfileId)
            updateActiveWizardProgress();

        Q_EMIT profilesChanged();
        if (normalizedProfile == m_activeProfileId)
            Q_EMIT activeProfileChanged();

        const bool persisted = persistProfiles();
        if (!persisted)
            qCWarning(lcUserProfiles) << "Failed to persist wizard progress step";
        return persisted;
    }

    return false;
}

bool UserProfileController::markWizardCompleted(const QString& profileId, bool completed)
{
    const QString normalizedProfile = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    if (normalizedProfile.isEmpty())
        return false;

    for (int i = 0; i < m_profiles.size(); ++i) {
        QVariantMap map = m_profiles.at(i).toMap();
        if (map.value(QStringLiteral("id")).toString() != normalizedProfile)
            continue;

        QVariantMap progress = sanitizedWizardProgress(map.value(QStringLiteral("setupProgress")).toMap());
        const bool previousCompleted = progress.value(QStringLiteral("completed")).toBool();
        if (previousCompleted == completed)
            return true;

        progress.insert(QStringLiteral("completed"), completed);
        progress.insert(QStringLiteral("updatedAt"), timestampUtc());
        map.insert(QStringLiteral("setupProgress"), progress);
        map.insert(QStringLiteral("updatedAt"), timestampUtc());
        m_profiles[i] = map;

        if (normalizedProfile == m_activeProfileId)
            updateActiveWizardProgress();

        Q_EMIT profilesChanged();
        if (normalizedProfile == m_activeProfileId)
            Q_EMIT activeProfileChanged();

        const bool persisted = persistProfiles();
        if (!persisted)
            qCWarning(lcUserProfiles) << "Failed to persist wizard completion";
        return persisted;
    }

    return false;
}

bool UserProfileController::resetWizardProgress(const QString& profileId)
{
    const QString normalizedProfile = normalizeProfileId(profileId.isEmpty() ? m_activeProfileId : profileId);
    if (normalizedProfile.isEmpty())
        return false;

    const QVariantMap defaults = defaultWizardProgress();
    for (int i = 0; i < m_profiles.size(); ++i) {
        QVariantMap map = m_profiles.at(i).toMap();
        if (map.value(QStringLiteral("id")).toString() != normalizedProfile)
            continue;

        QVariantMap progress = sanitizedWizardProgress(map.value(QStringLiteral("setupProgress")).toMap());
        if (progress == defaults)
            return true;

        map.insert(QStringLiteral("setupProgress"), defaults);
        map.insert(QStringLiteral("updatedAt"), timestampUtc());
        m_profiles[i] = map;

        if (normalizedProfile == m_activeProfileId)
            updateActiveWizardProgress();

        Q_EMIT profilesChanged();
        if (normalizedProfile == m_activeProfileId)
            Q_EMIT activeProfileChanged();

        const bool persisted = persistProfiles();
        if (!persisted)
            qCWarning(lcUserProfiles) << "Failed to reset wizard progress";
        return persisted;
    }

    return false;
}

void UserProfileController::setCatalogSnapshot(const QVariantList& definitions)
{
    m_catalogDefinitions = definitions;
    updateCatalogStrategyCache(definitions);
    updateActiveRecommendations();
}

QVariantMap UserProfileController::buildDefaultProfile(const QString& id) const
{
    QVariantMap profile;
    profile.insert(QStringLiteral("id"), id);
    profile.insert(QStringLiteral("displayName"), tr("Domyślny profil"));
    profile.insert(QStringLiteral("theme"), QStringLiteral("midnight"));
    profile.insert(QStringLiteral("favorites"), QStringList());
    profile.insert(QStringLiteral("paletteOverrides"), QVariantMap());
    profile.insert(QStringLiteral("setupProgress"), defaultWizardProgress());
    profile.insert(QStringLiteral("createdAt"), timestampUtc());
    profile.insert(QStringLiteral("updatedAt"), profile.value(QStringLiteral("createdAt")));
    return profile;
}

QString UserProfileController::normalizeProfileId(const QString& candidate) const
{
    QString lowered = candidate.trimmed().toLower();
    if (lowered.isEmpty())
        return {};
    static const QRegularExpression invalidChars(QStringLiteral("[^a-z0-9_\\-]+"));
    lowered.replace(invalidChars, QStringLiteral("-"));
    while (lowered.contains(QStringLiteral("--")))
        lowered.replace(QStringLiteral("--"), QStringLiteral("-"));
    if (lowered.startsWith(QLatin1Char('-')))
        lowered.remove(0, 1);
    if (lowered.endsWith(QLatin1Char('-')))
        lowered.chop(1);
    if (lowered.isEmpty())
        return {};
    return lowered;
}

QVariantMap UserProfileController::sanitizedProfile(const QVariantMap& profile, const QString& skipId) const
{
    QVariantMap result = profile;
    QString normalizedSkip = normalizeProfileId(skipId);

    QString id = normalizeProfileId(profile.value(QStringLiteral("id")).toString());
    if (id.isEmpty())
        id = normalizeProfileId(profile.value(QStringLiteral("displayName")).toString());
    if (id.isEmpty())
        id = QStringLiteral("profile");
    id = ensureUniqueProfileId(id, normalizedSkip);
    result.insert(QStringLiteral("id"), id);

    QString displayName = profile.value(QStringLiteral("displayName"), id).toString().trimmed();
    if (displayName.isEmpty())
        displayName = id;
    result.insert(QStringLiteral("displayName"), displayName);

    QString theme = profile.value(QStringLiteral("theme"), QStringLiteral("midnight")).toString();
    if (!m_themePalettes.contains(theme))
        theme = QStringLiteral("midnight");
    result.insert(QStringLiteral("theme"), theme);

    QStringList favorites;
    const QVariant favVariant = profile.value(QStringLiteral("favorites"));
    if (favVariant.canConvert<QStringList>()) {
        const QStringList rawList = favVariant.toStringList();
        for (const QString& entry : rawList) {
            const QString trimmed = entry.trimmed();
            if (trimmed.isEmpty())
                continue;
            if (!favorites.contains(trimmed)) {
                if (!m_catalogStrategyIds.isEmpty() && !m_catalogStrategyIds.contains(trimmed))
                    continue;
                favorites.append(trimmed);
            }
        }
    }
    result.insert(QStringLiteral("favorites"), favorites);

    const QVariant overridesVariant = profile.value(QStringLiteral("paletteOverrides"));
    if (overridesVariant.canConvert<QVariantMap>()) {
        const QVariantMap sanitizedOverrides = sanitizedPaletteOverrides(overridesVariant.toMap());
        if (sanitizedOverrides.isEmpty())
            result.remove(QStringLiteral("paletteOverrides"));
        else
            result.insert(QStringLiteral("paletteOverrides"), sanitizedOverrides);
    } else {
        result.remove(QStringLiteral("paletteOverrides"));
    }

    const QVariant progressVariant = profile.value(QStringLiteral("setupProgress"));
    const QVariantMap sanitizedProgress = sanitizedWizardProgress(progressVariant.toMap());
    result.insert(QStringLiteral("setupProgress"), sanitizedProgress);

    return result;
}

bool UserProfileController::writeProfiles(const QVariantMap& payload) const
{
    if (m_profilesPath.isEmpty())
        return false;

    const QFileInfo info(m_profilesPath);
    QDir dir = info.dir();
    if (!dir.exists() && !dir.mkpath(QStringLiteral(".")))
        return false;

    QFile file(m_profilesPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
        return false;

    const QJsonDocument doc = QJsonDocument::fromVariant(payload);
    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();
    return true;
}

QVariantMap UserProfileController::readProfiles() const
{
    if (m_profilesPath.isEmpty())
        return {};

    QFile file(m_profilesPath);
    if (!file.exists() || !file.open(QIODevice::ReadOnly | QIODevice::Text))
        return {};

    const QByteArray data = file.readAll();
    file.close();

    QJsonParseError error{};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &error);
    if (error.error != QJsonParseError::NoError || !doc.isObject())
        return {};
    return doc.object().toVariantMap();
}

bool UserProfileController::ensureDefaultProfile()
{
    if (!m_profiles.isEmpty())
        return false;
    m_profiles.append(buildDefaultProfile(QStringLiteral("default")));
    m_activeProfileId = QStringLiteral("default");
    return true;
}

void UserProfileController::rebuildThemePalette()
{
    applyActiveProfileTheme();
    Q_EMIT themePaletteChanged();
}

void UserProfileController::updateActiveRecommendations()
{
    Q_EMIT catalogIntegrationChanged();
}

void UserProfileController::updateCatalogStrategyCache(const QVariantList& definitions)
{
    m_catalogStrategyIds.clear();
    for (const QVariant& entry : definitions) {
        const QVariantMap map = entry.toMap();
        const QString name = map.value(QStringLiteral("name")).toString().trimmed();
        if (!name.isEmpty())
            m_catalogStrategyIds.insert(name);
    }
}

bool UserProfileController::updateAvailableThemes()
{
    bool changed = false;
    const QString defaultTheme = QStringLiteral("midnight");
    for (int i = 0; i < m_profiles.size(); ++i) {
        QVariantMap map = m_profiles.at(i).toMap();
        const QString theme = map.value(QStringLiteral("theme"), defaultTheme).toString();
        if (!m_themePalettes.contains(theme)) {
            map.insert(QStringLiteral("theme"), defaultTheme);
            m_profiles[i] = map;
            changed = true;
        }
    }
    return changed;
}

bool UserProfileController::setActiveProfileInternal(const QString& profileId,
                                                     bool emitSignals,
                                                     bool* changed)
{
    const QString normalized = normalizeProfileId(profileId);
    if (normalized.isEmpty())
        return false;

    bool exists = false;
    for (const QVariant& entry : m_profiles) {
        if (entry.toMap().value(QStringLiteral("id")).toString() == normalized) {
            exists = true;
            break;
        }
    }

    if (!exists)
        return false;

    const bool changedLocal = (m_activeProfileId != normalized);
    m_activeProfileId = normalized;
    applyActiveProfileTheme();
    updateActiveRecommendations();
    updateActiveWizardProgress();

    if (changed)
        *changed = changedLocal;

    if (emitSignals && changedLocal)
        Q_EMIT activeProfileChanged();

    return true;
}

bool UserProfileController::persistProfiles()
{
    QVariantMap payload;
    payload.insert(QStringLiteral("active"), m_activeProfileId);
    payload.insert(QStringLiteral("profiles"), m_profiles);
    const bool ok = writeProfiles(payload);
    if (!ok)
        qCWarning(lcUserProfiles) << "Failed to write user profiles to" << m_profilesPath;
    return ok;
}

bool UserProfileController::applyActiveProfileTheme()
{
    const QVariantMap profile = activeProfile();
    QString theme = profile.value(QStringLiteral("theme"), QStringLiteral("midnight")).toString();
    if (!m_themePalettes.contains(theme))
        theme = QStringLiteral("midnight");
    QVariantMap palette = m_themePalettes.value(theme, m_themePalettes.value(QStringLiteral("midnight")));
    const QVariantMap overrides = sanitizedPaletteOverrides(profile.value(QStringLiteral("paletteOverrides")).toMap());
    for (auto it = overrides.constBegin(); it != overrides.constEnd(); ++it)
        palette.insert(it.key(), it.value());

    if (m_activeThemePalette == palette)
        return true;
    m_activeThemePalette = palette;
    Q_EMIT themePaletteChanged();
    return true;
}

QString UserProfileController::ensureUniqueProfileId(const QString& baseId, const QString& skipId) const
{
    const QString normalizedSkip = normalizeProfileId(skipId);
    QString normalizedBase = normalizeProfileId(baseId);
    if (normalizedBase.isEmpty())
        normalizedBase = QStringLiteral("profile");

    QString candidate = normalizedBase;
    int suffix = 2;
    while (profileIdExists(candidate, normalizedSkip)) {
        candidate = QStringLiteral("%1_%2").arg(normalizedBase).arg(suffix++);
    }
    return candidate;
}

bool UserProfileController::profileIdExists(const QString& id, const QString& skipId) const
{
    const QString normalizedId = normalizeProfileId(id);
    if (normalizedId.isEmpty())
        return false;
    const QString normalizedSkip = normalizeProfileId(skipId);

    for (const QVariant& entry : m_profiles) {
        const QVariantMap map = entry.toMap();
        const QString existingId = normalizeProfileId(map.value(QStringLiteral("id")).toString());
        if (existingId.isEmpty())
            continue;
        if (existingId == normalizedId) {
            if (!normalizedSkip.isEmpty() && existingId == normalizedSkip)
                continue;
            return true;
        }
    }
    return false;
}

QVariantMap UserProfileController::sanitizedWizardProgress(const QVariantMap& progress) const
{
    QVariantMap sanitized = defaultWizardProgress();

    QStringList steps;
    const QVariant stepsVariant = progress.value(QStringLiteral("completedSteps"));
    if (stepsVariant.canConvert<QStringList>()) {
        const QStringList rawSteps = stepsVariant.toStringList();
        for (const QString& entry : rawSteps) {
            const QString trimmed = entry.trimmed();
            if (trimmed.isEmpty())
                continue;
            if (!steps.contains(trimmed))
                steps.append(trimmed);
        }
    }
    sanitized.insert(QStringLiteral("completedSteps"), steps);

    const bool completed = progress.value(QStringLiteral("completed")).toBool();
    sanitized.insert(QStringLiteral("completed"), completed);

    const QString lastStep = progress.value(QStringLiteral("lastStep")).toString().trimmed();
    if (!lastStep.isEmpty())
        sanitized.insert(QStringLiteral("lastStep"), lastStep);

    const QString updatedAt = progress.value(QStringLiteral("updatedAt")).toString().trimmed();
    if (!updatedAt.isEmpty())
        sanitized.insert(QStringLiteral("updatedAt"), updatedAt);

    return sanitized;
}

QVariantMap UserProfileController::defaultWizardProgress() const
{
    QVariantMap progress;
    progress.insert(QStringLiteral("completedSteps"), QStringList());
    progress.insert(QStringLiteral("completed"), false);
    return progress;
}

QVariantMap UserProfileController::sanitizedPaletteOverrides(const QVariantMap& overrides) const
{
    QVariantMap sanitized;
    for (auto it = overrides.constBegin(); it != overrides.constEnd(); ++it) {
        const QString key = it.key();
        if (!m_paletteRoles.contains(key))
            continue;
        const QString normalizedColor = normalizeColorCode(it.value().toString());
        if (normalizedColor.isEmpty() || !isValidColorCode(normalizedColor))
            continue;
        sanitized.insert(key, normalizedColor);
    }
    return sanitized;
}

bool UserProfileController::isValidColorCode(const QString& color) const
{
    static const QRegularExpression hexPattern(QStringLiteral("^#(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$"));
    return hexPattern.match(color).hasMatch();
}

QString UserProfileController::normalizeColorCode(const QString& color) const
{
    QString trimmed = color.trimmed();
    if (trimmed.isEmpty())
        return {};
    if (!trimmed.startsWith(QLatin1Char('#')))
        trimmed.prepend(QLatin1Char('#'));
    return trimmed.toUpper();
}

void UserProfileController::updateActiveWizardProgress()
{
    const QVariantMap next = wizardProgress(m_activeProfileId);
    if (m_activeWizardProgress == next)
        return;
    m_activeWizardProgress = next;
    Q_EMIT wizardProgressChanged();
}

QVariantList UserProfileController::collectSortedDefinitions(const QVariantList& definitions,
                                                             const QStringList& exclude) const
{
    struct RankedDefinition {
        QVariantMap map;
        double score = 0.0;
        QString name;
    };

    QList<RankedDefinition> ranked;
    for (const QVariant& entry : definitions) {
        const QVariantMap map = entry.toMap();
        const QString name = map.value(QStringLiteral("name")).toString().trimmed();
        if (name.isEmpty())
            continue;
        if (exclude.contains(name))
            continue;
        RankedDefinition rd;
        rd.map = map;
        rd.name = name;
        const QVariant metaVariant = map.value(QStringLiteral("metadata"));
        if (metaVariant.canConvert<QVariantMap>()) {
            const QVariantMap metadata = metaVariant.toMap();
            const QVariant scoreVariant = metadata.value(QStringLiteral("popularity"), metadata.value(QStringLiteral("score")));
            bool ok = false;
            const double score = scoreVariant.toDouble(&ok);
            if (ok)
                rd.score = score;
        }
        ranked.append(rd);
    }

    std::sort(ranked.begin(), ranked.end(), [](const RankedDefinition& lhs, const RankedDefinition& rhs) {
        if (!qFuzzyCompare(lhs.score + 1.0, rhs.score + 1.0))
            return lhs.score > rhs.score;
        return lhs.name < rhs.name;
    });

    QVariantList result;
    for (const RankedDefinition& item : ranked) {
        result.append(item.map);
        if (result.size() >= 6)
            break;
    }
    return result;
}
